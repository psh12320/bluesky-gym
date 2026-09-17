import copy
import csv
import json
from pathlib import Path
import pytest
from atc_rl.matrix import SEEDS,SUPPORTS,sha
from atc_rl.matrix_evaluate import (task,select_checkpoint,validate_training,verify_evaluation,
                                    aggregate_seeds,support_effects)


def write(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def test_evaluation_array_keeps_every_seed_support_and_registered_curve_point():
    rows=[{'seed':s,'guidance':g,'filter':f} for g,f in SUPPORTS for s in SEEDS]
    expected={(s,g,f,t) for s in SEEDS for g,f in SUPPORTS for t in ('initial','100k','300k','final')}
    selected=[task({'rows':rows},i) for i in range(48)]
    assert {(r['seed'],r['guidance'],r['filter'],t) for r,t in selected}==expected
    assert task({'rows':rows},48)==(None,'classical')
    for index in (-1,49,True,1.5):
        with pytest.raises(ValueError):task({'rows':rows},index)


def checkpoints(directory):
    records=[]
    for live,name in [(0,'initial-model.zip'),(101234,'policy-live-101234.zip'),
                      (303456,'policy-live-303456.zip'),(1000234,'model.zip')]:
        path=directory/name;path.write_bytes(str(live).encode())
        records.append(dict(file=name,sha256=sha(path),live_transitions=live,
                            counted_transitions=live,optimizer_steps=int(live>0)))
    write(directory/'checkpoints.json',records)
    write(directory/'config.json',dict(workers=2,rollout_steps=256))
    write(directory/'training_summary.json',dict(status='complete',live_transitions=1000234,model_sha256=records[-1]['sha256']))
    return records


def test_curve_selection_retains_actual_live_counts_and_rejects_distant_points(tmp_path):
    records=checkpoints(tmp_path)
    assert [select_checkpoint(tmp_path,t)[1]['live_transitions'] for t in ('initial','100k','300k','final')]==[0,101234,303456,1000234]
    write(tmp_path/'checkpoints.json',[records[0],records[2],records[3]])
    with pytest.raises(ValueError,match='too far'):select_checkpoint(tmp_path,'100k')


@pytest.mark.parametrize('failure',['duplicate','changed_file','trained_initial','partial_final','changed_final_count'])
def test_curve_checkpoint_integrity_and_budget_failures_are_rejected(tmp_path,failure):
    records=checkpoints(tmp_path);stage='initial'
    if failure=='duplicate':records.append(records[0]);write(tmp_path/'checkpoints.json',records)
    elif failure=='changed_file':(tmp_path/'initial-model.zip').write_bytes(b'changed')
    elif failure=='trained_initial':records[0]['optimizer_steps']=1;write(tmp_path/'checkpoints.json',records)
    else:
        stage='final';summary=json.loads((tmp_path/'training_summary.json').read_text())
        if failure=='partial_final':summary['status']='wall_limit_before_budget'
        else:summary['live_transitions']+=1
        write(tmp_path/'training_summary.json',summary)
    with pytest.raises(ValueError):select_checkpoint(tmp_path,stage)


def completed_training(root):
    directory=root/'runs/row';directory.mkdir(parents=True)
    config={'live_steps':1000000,'reward_scale':.01}
    row={'run_dir':'runs/row','algorithm':'ppo','seed':50100,'guidance':False,'filter':True}
    plan={'config':config,'source_sha256':{'atc_rl/train.py':'abc','pyproject.toml':'def','jobs/train.slurm':'xyz'}}
    write(directory/'config.json',{**config,**row})
    summary=dict(status='complete',live_transitions=1000200,optimizer_steps=234,model_sha256='model')
    write(directory/'training_summary.json',summary);write(directory/'checkpoint-audit.json',summary)
    write(directory/'provenance.json',{'source_sha256':{'atc_rl/train.py':'abc','pyproject.toml':'def'}})
    return plan,row,directory


@pytest.mark.parametrize('failure',[None,'partial','under_budget','recipe','audit','source'])
def test_evaluations_require_completed_matched_training_and_provenance(tmp_path,failure):
    plan,row,directory=completed_training(tmp_path)
    if failure in ('partial','under_budget'):
        path=directory/'training_summary.json';data=json.loads(path.read_text())
        if failure=='partial':data['status']='wall_limit_before_budget'
        else:data['live_transitions']=999999
    elif failure=='recipe':path=directory/'config.json';data=json.loads(path.read_text());data['reward_scale']=1.
    elif failure=='audit':path=directory/'checkpoint-audit.json';data=json.loads(path.read_text());data['optimizer_steps']=0
    elif failure=='source':path=directory/'provenance.json';data={'source_sha256':{}}
    if failure:
        write(path,data)
        with pytest.raises(ValueError):validate_training(tmp_path,plan,row)
    else:assert validate_training(tmp_path,plan,row)[0]==directory


def evaluation(root):
    from atc.metrics import METRICS,summarize
    directory=root/'evaluation';directory.mkdir()
    records=[]
    for episode in range(20):
        for agent in range(10):
            records.append({'episode':episode,'agent':str(agent),'scenario_sha256':str(episode),
                            **{m:0. for m in METRICS},'waypoint_reached':1.,'flight_time':1000.})
    path=directory/'aircraft.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary=summarize(records,20,10);summary['csv_sha256']=sha(path);write(directory/'summary.json',summary)
    protocol=dict(seed=20260,episodes=20,algorithm='classical',guidance=True,filter=True,
                  action_reference='direct',checkpoint=None,model_path=None,source_sha256={'atc_rl/evaluate.py':'abc'})
    write(directory/'protocol.json',protocol)
    plan={'source_sha256':{**protocol['source_sha256'],'jobs/evaluate.slurm':'def'}}
    return directory,plan,protocol


@pytest.mark.parametrize('failure',[None,'csv','summary','stream','source','scaled','filter','checkpoint'])
def test_reusing_evaluations_verifies_physical_results_and_protocol(tmp_path,failure):
    directory,plan,protocol=evaluation(tmp_path)
    if failure=='csv':
        with (directory/'aircraft.csv').open('a') as stream:stream.write('changed')
    elif failure=='summary':
        path=directory/'summary.json';summary=json.loads(path.read_text());summary['metrics']['flight_time']['mean']=999.;write(path,summary)
    elif failure:
        if failure=='stream':protocol['seed']=20301
        elif failure=='source':protocol['source_sha256']={}
        elif failure=='scaled':protocol['evaluation_reward_scale']=.01
        elif failure=='filter':protocol['filter']=False
        elif failure=='checkpoint':protocol['checkpoint']={'file':'model.zip'}
        write(directory/'protocol.json',protocol)
    if failure:
        with pytest.raises(ValueError):verify_evaluation(tmp_path,plan,directory,None,'classical')
    else:assert len(verify_evaluation(tmp_path,plan,directory,None,'classical')[2])==20


def seed_rows():
    rows=[]
    for g,f in SUPPORTS:
        for seed,base_change in zip(SEEDS,[1.,2.,-6.]):
            initial=10.+100*g+20*f
            change=base_change+3*g+5*f+7*g*f
            rows.append(dict(algorithm='ppo',seed=seed,guidance=g,filter=f,
                             initial_means={'metric':initial},final_means={'metric':initial+change}))
    return rows


def test_seed_aggregation_keeps_negative_seed_and_separates_prior_from_learning():
    rows=seed_rows();groups=aggregate_seeds(rows)
    assert groups[0]['metrics']['metric']['mean_learning_change']==-1.
    assert groups[0]['metrics']['metric']['per_seed_learning_changes']=={'50100':1.,'50200':2.,'50300':-6.}
    assert groups[0]['metrics']['metric']['learning_change_sample_standard_deviation']==pytest.approx(19.**.5)
    effects=support_effects(rows)
    g=effects['guidance_with_filter_off']['metric']
    assert g['initial_support_effect']['mean']==100.
    assert g['final_support_effect']['mean']==103.
    assert g['difference_in_learning_changes']['mean']==3.
    assert effects['filter_with_guidance_off']['metric']['difference_in_learning_changes']['mean']==5.
    assert effects['guidance_filter_interaction']['metric']['difference_in_learning_changes']['mean']==7.


@pytest.mark.parametrize('failure',['missing','duplicate','extra','mixed_algorithm','nonfinite','missing_metric'])
def test_seed_aggregation_rejects_cherry_picked_or_incompatible_rows(failure):
    rows=seed_rows()
    if failure=='missing':rows.pop()
    elif failure=='duplicate':rows[2]=copy.deepcopy(rows[0])
    elif failure=='extra':rows.append(copy.deepcopy(rows[0]))
    elif failure=='mixed_algorithm':rows[0]['algorithm']='mappo'
    elif failure=='nonfinite':rows[0]['final_means']['metric']=float('nan')
    else:rows[0]['final_means']={}
    with pytest.raises(ValueError):aggregate_seeds(rows)
