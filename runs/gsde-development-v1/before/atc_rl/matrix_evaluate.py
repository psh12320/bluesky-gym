"""Evaluate every matrix checkpoint and aggregate learning without hiding seeds."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from atc_rl.matrix import checked_path,load_plan,read,sha
from atc_rl.checkpoint_identity import verified_checkpoint

STAGES=('initial','100k','300k','final')
EVALUATION_SEED=20260
EPISODES=20


def task(plan,index):
    count=len(plan['rows'])*len(STAGES)
    if type(index) is not int or not 0<=index<=count:raise ValueError('Evaluation array index outside matrix')
    if index==count:return None,'classical'
    return plan['rows'][index//len(STAGES)],STAGES[index%len(STAGES)]


def output_directory(root,plan_path,row,stage):
    if stage=='classical':return checked_path(root,plan_path).parent/'classical-dev20'
    return checked_path(root,row['run_dir'])/'evaluations'/('dev20-'+stage)


def validate_training(root,plan,row):
    directory=checked_path(root,row['run_dir']);config=read(directory/'config.json')
    if config.get('static_filter',False):raise ValueError('Existing matrix protocol does not include static_filter')
    if config.get('conflict_features',False) or config.get('mask_conflict_features',False):raise ValueError('Existing matrix protocol does not include conflict_features')
    summary=read(directory/'training_summary.json');audit=read(directory/'checkpoint-audit.json')
    if summary['status']!='complete' or summary['live_transitions']<plan['config']['live_steps']:
        raise ValueError('Matrix evaluation requires completed training at the requested budget')
    for key,value in {**plan['config'],**{k:row[k] for k in ('algorithm','seed','guidance','filter')}}.items():
        if config.get(key)!=value:raise ValueError('Training differs from matrix: '+key)
    if audit['status']!='complete' or any(audit[k]!=summary[k] for k in ('live_transitions','optimizer_steps','model_sha256')):
        raise ValueError('Training audit differs from the completed run')
    expected={k:v for k,v in plan['source_sha256'].items() if k.endswith('.py') or k=='pyproject.toml'}
    if read(directory/'provenance.json')['source_sha256']!=expected:
        raise ValueError('Training provenance differs from the frozen matrix source')
    return directory,summary


def select_checkpoint(directory,stage):
    manifest=read(directory/'checkpoints.json');config=read(directory/'config.json')
    if len({r['file'] for r in manifest})!=len(manifest):raise ValueError('Duplicate checkpoint filenames')
    if stage in ('initial','final'):
        filename='initial-model.zip' if stage=='initial' else 'model.zip'
        choices=[r for r in manifest if r['file']==filename]
        if len(choices)!=1:raise ValueError('Missing unique '+stage+' checkpoint')
        record=choices[0]
        if stage=='initial' and any(record[k]!=0 for k in ('live_transitions','counted_transitions','optimizer_steps')):
            raise ValueError('Initial checkpoint already contains training')
        if stage=='final':
            summary=read(directory/'training_summary.json')
            if summary['status']!='complete' or record['live_transitions']!=summary['live_transitions'] or record['sha256']!=summary['model_sha256']:
                raise ValueError('Final checkpoint differs from the completed training summary')
    elif stage in ('100k','300k'):
        threshold=100000 if stage=='100k' else 300000
        choices=[r for r in manifest if r['file'].startswith('policy-live-') and r['live_transitions']>=threshold]
        if not choices:raise ValueError('No saved checkpoint at '+stage)
        record=min(choices,key=lambda r:r['live_transitions'])
        if record['live_transitions']>threshold+config['workers']*10*config['rollout_steps']:
            raise ValueError('Saved checkpoint is too far from the requested experience point')
    else:raise ValueError('Unknown checkpoint stage')
    return verified_checkpoint(directory,record),record


def verify_evaluation(root,plan,directory,row,stage,record=None,checkpoint=None):
    from atc_rl.compare import load_evaluation
    loaded=load_evaluation(directory);summary,protocol,scenarios,_=loaded
    if summary['episodes']!=EPISODES or summary['agent_episodes']!=10*EPISODES or len(scenarios)!=EPISODES:
        raise ValueError('Incomplete matrix evaluation')
    if protocol['seed']!=EVALUATION_SEED or protocol['episodes']!=EPISODES:raise ValueError('Wrong development stream or budget')
    if protocol.get('static_filter',False):raise ValueError('Existing matrix protocol does not include static_filter')
    if protocol.get('conflict_features',False) or protocol.get('mask_conflict_features',False):raise ValueError('Existing matrix protocol does not include conflict_features')
    expected_sources={k:v for k,v in plan['source_sha256'].items() if k.endswith('.py')}
    if protocol['source_sha256']!=expected_sources:raise ValueError('Evaluation source differs from matrix')
    if protocol.get('evaluation_reward_scale',1.)!=1. or protocol.get('evaluation_progress_scale',0.)!=0.:
        raise ValueError('Matrix requires native scoring rewards')
    expected={'algorithm':'classical','guidance':True,'filter':True,'action_reference':'direct'} if row is None else {
        **{k:row[k] for k in ('algorithm','guidance','filter')},'action_reference':plan['config']['action_reference']}
    for key,value in expected.items():
        if protocol.get(key,'direct' if key=='action_reference' else None)!=value:raise ValueError('Evaluation changed '+key)
    if row is None:
        if protocol['checkpoint'] is not None:raise ValueError('Classical reference unexpectedly uses a learned checkpoint')
    elif protocol['checkpoint']!=record or Path(protocol['model_path']).resolve()!=checkpoint.resolve():
        raise ValueError('Evaluation used the wrong training checkpoint')
    return loaded


def run_evaluation(root,plan_path,index,dry_run=False):
    plan=load_plan(root,plan_path);row,stage=task(plan,index)
    directory=output_directory(root,plan_path,row,stage)
    if dry_run:
        return {'row':row,'stage':stage,'output':str(directory),'evaluation_started':False,
                'checkpoint_selection':'Verified saved manifest at launch; actual live count is retained.',
                'classical_reference_reuse':'Only after complete metrics, source and scenario-protocol checks.'}
    checkpoint=record=None
    if row is not None:
        training,_=validate_training(root,plan,row);checkpoint,record=select_checkpoint(training,stage)
    if directory.exists():
        # Completed immutable evaluations can be reused; incomplete outputs are not overwritten.
        verify_evaluation(root,plan,directory,row,stage,record,checkpoint)
        return {'stage':stage,'output':str(directory),'reused_verified_evaluation':True}
    plan_file=checked_path(root,plan_path)
    receipt_name='classical.json' if row is None else f'{plan["algorithm"]}-task-{index:02d}.json'
    receipt=plan_file.parent/'evaluation-launches'/receipt_name
    receipt.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-m','atc_rl.evaluate','--episodes',str(EPISODES),'--seed',str(EVALUATION_SEED),'--out',str(directory)]
    command+=['--classical'] if row is None else ['--model',str(checkpoint)]
    with receipt.open('x',encoding='utf-8') as stream:
        json.dump({'row':row,'stage':stage,'checkpoint':record,'plan_sha256':sha(plan_file),'command':command},stream,indent=2)
    subprocess.run(command,cwd=root,check=True)
    load_plan(root,plan_path)
    verify_evaluation(root,plan,directory,row,stage,record,checkpoint)
    return {'stage':stage,'output':str(directory),'reused_verified_evaluation':False}


def aggregate_seeds(rows):
    """Show all three seed effects; do not pool aircraft as independent replications."""
    import numpy as np
    from atc_rl.matrix import SUPPORTS,SEEDS
    if len(rows)!=len(SUPPORTS)*len(SEEDS):raise ValueError('Expected the complete twelve-row matrix')
    if len({r['algorithm'] for r in rows})!=1:raise ValueError('Do not mix algorithms in seed aggregation')
    metric_names=set(rows[0]['initial_means'])
    if not metric_names:raise ValueError('Missing performance metrics')
    for row in rows:
        for field in ('initial_means','final_means'):
            if set(row[field])!=metric_names or not np.isfinite(list(row[field].values())).all():
                raise ValueError('Missing or non-finite performance metrics')
    groups=[]
    for guidance,filtered in SUPPORTS:
        selected=[r for r in rows if r['guidance']==guidance and r['filter']==filtered]
        if sorted(r['seed'] for r in selected)!=list(SEEDS):raise ValueError('All three unique training seeds are required')
        metrics={}
        for metric in selected[0]['initial_means']:
            before=np.array([r['initial_means'][metric] for r in selected],dtype=float)
            after=np.array([r['final_means'][metric] for r in selected],dtype=float);delta=after-before
            metrics[metric]={'initial_mean':float(before.mean()),'final_mean':float(after.mean()),'mean_learning_change':float(delta.mean()),
                'learning_change_sample_standard_deviation':float(delta.std(ddof=1)),
                'per_seed_learning_changes':{str(r['seed']):float(d) for r,d in zip(selected,delta)}}
        groups.append({'guidance':guidance,'filter':filtered,'training_seeds':3,'metrics':metrics})
    return groups


def support_effects(rows):
    """Separate support effects from changes caused by training in each setting."""
    import numpy as np
    from atc_rl.matrix import SEEDS
    aggregate_seeds(rows)
    lookup={(r['seed'],r['guidance'],r['filter']):r for r in rows}
    contrasts={
        'guidance_with_filter_off':[(True,False,1),(False,False,-1)],
        'guidance_with_filter_on':[(True,True,1),(False,True,-1)],
        'filter_with_guidance_off':[(False,True,1),(False,False,-1)],
        'filter_with_guidance_on':[(True,True,1),(True,False,-1)],
        'guidance_filter_interaction':[(True,True,1),(True,False,-1),(False,True,-1),(False,False,1)],
    }
    results={}
    for name,terms in contrasts.items():
        results[name]={}
        for metric in rows[0]['initial_means']:
            by_seed={}
            for seed in SEEDS:
                before=sum(sign*lookup[seed,g,f]['initial_means'][metric] for g,f,sign in terms)
                after=sum(sign*lookup[seed,g,f]['final_means'][metric] for g,f,sign in terms)
                by_seed[str(seed)]={'initial_support_effect':before,'final_support_effect':after,
                                    'difference_in_learning_changes':after-before}
            results[name][metric]={'per_seed':by_seed}
            for measure in next(iter(by_seed.values())):
                values=np.array([item[measure] for item in by_seed.values()])
                results[name][metric][measure]={'mean':float(values.mean()),'sample_standard_deviation':float(values.std(ddof=1))}
    return results


def analyze(root,plan_path,out):
    plan=load_plan(root,plan_path);destination=checked_path(root,out)
    if destination.exists():raise ValueError('Choose a fresh matrix analysis directory')
    classical=output_directory(root,plan_path,None,'classical')
    reference=verify_evaluation(root,plan,classical,None,'classical')
    validated=[]
    for row in plan['rows']:
        training,_=validate_training(root,plan,row);evaluations={}
        for stage in STAGES:
            checkpoint,record=select_checkpoint(training,stage)
            directory=output_directory(root,plan_path,row,stage)
            loaded=verify_evaluation(root,plan,directory,row,stage,record,checkpoint)
            if loaded[2]!=reference[2]:raise ValueError('Matrix scenarios are not paired')
            evaluations[stage]=directory
        validated.append((row,training,evaluations))
    destination.mkdir(parents=True);rows=[]
    for row,training,evaluations in validated:
        folder=destination/f'row-{row["index"]:02d}';folder.mkdir()
        commands=[
            [sys.executable,'-m','atc_rl.compare','--initial',str(evaluations['initial']),'--trained',str(evaluations['final']),
             '--classical',str(classical),'--out',str(folder/'comparison')],
            [sys.executable,'-m','atc_rl.curves','--training-run',str(training),'--classical',str(classical),'--out',str(folder/'curve')]]
        for stage in STAGES:commands[1]+= ['--evaluation',str(evaluations[stage])]
        for index,command in enumerate(commands):
            result=subprocess.run(command,cwd=root,check=True,capture_output=True,text=True)
            (folder/f'analysis-{index}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        comparison=read(folder/'comparison/comparison.json')
        rows.append({**row,'checkpoint':comparison['checkpoint'],'initial_means':comparison['means']['initial'],
                     'final_means':comparison['means']['trained'],'comparison':str(folder/'comparison/comparison.json')})
    grouped=aggregate_seeds(rows)
    result={'algorithm':plan['algorithm'],'matrix_plan_sha256':sha(checked_path(root,plan_path)),'rows':rows,'groups':grouped,
        'support_effects':support_effects(rows),'classical_reference':str(classical),'evaluation_seed':EVALUATION_SEED,'development_worlds':EPISODES,
        'limitations':['Three training seeds; all seeds and failures must remain visible.',
            'Seed variability is reported as sample standard deviation, not a confidence interval from independent aircraft.',
            'Per-run paired-world intervals are in the individual comparisons; they do not measure training-seed uncertainty.',
            'Development results are not unseen-scenario or official competition evidence.',
            'The fixed classical system uses guidance/filtering; each matched initial/final pair measures learning within its support setting.']}
    (destination/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (destination/'seed-results.csv').open('x',newline='',encoding='utf-8') as stream:
        fields=['algorithm','seed','guidance','filter','live_transitions']
        metrics=list(rows[0]['initial_means']);fields += [prefix+metric for metric in metrics for prefix in ('initial_','final_','change_')]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            record={key:row[key] for key in ('algorithm','seed','guidance','filter')};record['live_transitions']=row['checkpoint']['live_transitions']
            for metric in metrics:
                record['initial_'+metric]=row['initial_means'][metric];record['final_'+metric]=row['final_means'][metric]
                record['change_'+metric]=row['final_means'][metric]-row['initial_means'][metric]
            writer.writerow(record)
    return {'analysis':str(destination),'training_seeds_per_support_setting':3,'completed_rows':len(rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='action',required=True)
    run=commands.add_parser('run');run.add_argument('--plan',type=Path,required=True);run.add_argument('--index',type=int,required=True);run.add_argument('--dry-run',action='store_true')
    summary=commands.add_parser('analyze');summary.add_argument('--plan',type=Path,required=True);summary.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=run_evaluation(root,args.plan,args.index,args.dry_run) if args.action=='run' else analyze(root,args.plan,args.out)
    print(json.dumps(result))


if __name__=='__main__':main()
