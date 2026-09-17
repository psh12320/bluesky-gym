import copy
import json
from pathlib import Path
import pytest
from atc_rl.matrix import (SEEDS,SUPPORTS,create_plan,load_plan,training_command,
                           validate_config,validate_reference,sha)


@pytest.fixture
def workspace(tmp_path):
    (tmp_path/'jobs').mkdir();(tmp_path/'jobs/requirements-onpolicy.txt').write_text('fixed dependencies')
    (tmp_path/'jobs/train_onpolicy_matrix.slurm').write_text('fixed job')
    (tmp_path/'pyproject.toml').write_text('fixed project')
    for package in ('atc','atc_rl','core','bluesky_gym','bluesky_zoo'):
        (tmp_path/package).mkdir();(tmp_path/package/'__init__.py').write_text('# fixture')
    return tmp_path


@pytest.fixture
def config():
    return dict(workers=8,live_steps=1000000,rollout_steps=256,batch_size=1024,epochs=10,
                action_reference='goal_offset',initial_action_std=.05,neutral_action_mean=True,
                reward_scale=.01,progress_scale=0.)


def pair(root,config):
    ppo_path=root/'runs/ppo/matrix.json';ppo=create_plan(root,ppo_path,config)
    mappo_path=root/'runs/mappo/matrix.json';mappo=create_plan(root,mappo_path,ppo_plan_path=ppo_path)
    return ppo_path,ppo,mappo_path,mappo


def test_every_support_pair_has_all_three_seeds_and_distinct_world_streams(workspace,config):
    path=workspace/'runs/ppo/matrix.json';plan=create_plan(workspace,path,config)
    assert load_plan(workspace,path)==plan
    assert len(plan['rows'])==12 and len({r['run_dir'] for r in plan['rows']})==12
    assert {(r['guidance'],r['filter'],r['seed']) for r in plan['rows']}=={(g,f,s) for g,f in SUPPORTS for s in SEEDS}
    streams=[{seed+10*w for w in range(config['workers'])} for seed in SEEDS]
    assert all(not a&b for i,a in enumerate(streams) for b in streams[i+1:])


def test_mappo_inherits_recipe_and_matches_each_ppo_seed_support_reference(workspace,config):
    _,ppo,path,mappo=pair(workspace,config)
    assert load_plan(workspace,path)==mappo and mappo['config']==ppo['config']
    for left,right in zip(ppo['rows'],mappo['rows']):
        assert right['actor_reference']==left['run_dir']+'/initial-model.zip'
        assert all(left[k]==right[k] for k in ('seed','guidance','filter'))
        command=training_command(workspace,right,mappo['config'],'cuda',9900.)
        assert command[command.index('--actor-reference')+1]==str(workspace/right['actor_reference'])
        assert command[command.index('--algorithm')+1]=='mappo'
        assert command[command.index('--action-reference')+1]=='goal_offset'
        assert ('--guidance' in command)==right['guidance'] and ('--filter' in command)==right['filter']


def test_matrix_rejects_changed_source_and_duplicate_or_reordered_rows(workspace,config):
    path=workspace/'runs/ppo/matrix.json';plan=create_plan(workspace,path,config)
    corrupted=copy.deepcopy(plan);corrupted['rows'][1]=corrupted['rows'][0]
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError,match='every support/seed'):load_plan(workspace,path)
    path.write_text(json.dumps(plan));(workspace/'core/__init__.py').write_text('changed')
    with pytest.raises(ValueError,match='source changed'):load_plan(workspace,path)


def test_mappo_rejects_changed_parent_plan(workspace,config):
    ppo_path,_,mappo_path,_=pair(workspace,config)
    with ppo_path.open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='reference plan changed'):load_plan(workspace,mappo_path)


@pytest.mark.parametrize('path',['../outside.json','runs/../outside.json','runs'])
def test_artifact_paths_cannot_escape_runs(workspace,config,path):
    with pytest.raises(ValueError,match='inside'):create_plan(workspace,workspace/path,config)


def test_existing_plan_is_not_overwritten(workspace,config):
    path=workspace/'runs/ppo/matrix.json';create_plan(workspace,path,config);before=path.read_bytes()
    with pytest.raises(ValueError,match='fresh'):create_plan(workspace,path,config)
    assert path.read_bytes()==before


def completed_reference(root,row,config):
    checkpoint=root/row['actor_reference'];checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'fixture checkpoint; no trained model')
    actual={**config,'algorithm':'ppo','seed':row['seed'],'guidance':row['guidance'],'filter':row['filter']}
    (checkpoint.parent/'config.json').write_text(json.dumps(actual))
    (checkpoint.parent/'training_summary.json').write_text(json.dumps({'status':'complete'}))
    (checkpoint.parent/'checkpoints.json').write_text(json.dumps([{'file':checkpoint.name,'sha256':sha(checkpoint),'live_transitions':0,'counted_transitions':0,'optimizer_steps':0}]))
    return checkpoint


def test_mappo_requires_completed_matching_ppo_and_calls_full_audit(workspace,config):
    *_,plan=pair(workspace,config);row=plan['rows'][4];checkpoint=completed_reference(workspace,row,config);calls=[]
    def auditor(run):calls.append(run);return {'status':'complete','checked_fixture':True}
    assert validate_reference(workspace,row,config,auditor)['checked_fixture']
    assert calls==[checkpoint.parent]
    (checkpoint.parent/'training_summary.json').write_text(json.dumps({'status':'wall_limit_before_budget'}))
    with pytest.raises(ValueError,match='did not complete'):validate_reference(workspace,row,config,auditor)
    assert len(calls)==1


@pytest.mark.parametrize('change',['seed','guidance','reward_scale','learned_actor','corrupt_actor'])
def test_mappo_rejects_mismatched_or_trained_actor_before_auditing(workspace,config,change):
    *_,plan=pair(workspace,config);row=plan['rows'][0];checkpoint=completed_reference(workspace,row,config)
    if change in ('seed','guidance','reward_scale'):
        path=checkpoint.parent/'config.json';value=json.loads(path.read_text());value[change]=not value[change] if change=='guidance' else value[change]+1
        path.write_text(json.dumps(value))
    elif change=='learned_actor':
        path=checkpoint.parent/'checkpoints.json';value=json.loads(path.read_text());value[0]['live_transitions']=1;path.write_text(json.dumps(value))
    else:checkpoint.write_bytes(b'changed')
    def auditor(_):raise AssertionError('Invalid reference reached full audit')
    with pytest.raises(ValueError):validate_reference(workspace,row,config,auditor)


def test_excess_workers_would_overlap_canonical_world_seed_streams(config):
    config['workers']=11
    with pytest.raises(ValueError,match='seed spacing'):validate_config(config)


def test_mappo_cli_rejects_silently_ignored_recipe_overrides(monkeypatch,capsys):
    from atc_rl.matrix import main
    monkeypatch.setattr('sys.argv',['matrix','create','--plan','runs/mappo.json','--ppo-plan','runs/ppo.json','--reward-scale','0.5'])
    with pytest.raises(SystemExit) as error:main()
    assert error.value.code==2
    assert 'omit recipe overrides' in capsys.readouterr().err
