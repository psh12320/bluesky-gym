"""Plan and execute the three-seed, four-support PPO/MAPPO comparison on Slurm."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SEEDS=(50100,50200,50300)
SUPPORTS=((False,False),(False,True),(True,False),(True,True))
CONFIG_KEYS={'workers','live_steps','rollout_steps','batch_size','epochs','action_reference',
             'initial_action_std','neutral_action_mean','reward_scale','progress_scale'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_hashes(root):
    files=[root/'pyproject.toml',root/'jobs/requirements-onpolicy.txt']
    files.extend((root/'jobs').glob('*.slurm'))
    for package in ('atc','atc_rl','core','bluesky_gym','bluesky_zoo'):
        files.extend((root/package).rglob('*.py'))
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(set(files))}


def checked_path(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root/'runs') or path==root/'runs':
        raise ValueError('Matrix artifacts must stay inside this source root runs directory')
    return path


def validate_config(config):
    if set(config)!=CONFIG_KEYS:raise ValueError('Unexpected matrix configuration fields')
    for name in ('workers','live_steps','rollout_steps','batch_size','epochs'):
        if type(config[name]) is not int or config[name]<1:raise ValueError('Invalid '+name)
    if config['workers']>10:raise ValueError('Canonical seed spacing permits at most ten workers')
    if config['action_reference'] not in ('direct','goal_offset'):raise ValueError('Unknown action reference')
    if type(config['neutral_action_mean']) is not bool:raise ValueError('Expected a neutral-mean boolean')
    for name in ('initial_action_std','reward_scale','progress_scale'):
        if not isinstance(config[name],(int,float)) or not math.isfinite(config[name]):raise ValueError('Invalid '+name)
    if not 0<config['initial_action_std']<=1 or config['reward_scale']<=0 or config['progress_scale']<0:
        raise ValueError('Invalid action noise or reward transformation')


def expected_rows(root,plan_path,algorithm,config,ppo_plan=None):
    rows=[]
    for guidance,filtered in SUPPORTS:
        for seed in SEEDS:
            index=len(rows)
            run=plan_path.parent/f'{algorithm}-g{int(guidance)}-f{int(filtered)}-seed{seed}'
            row={'index':index,'algorithm':algorithm,'seed':seed,'guidance':guidance,'filter':filtered,
                 'run_dir':run.relative_to(root).as_posix()}
            if ppo_plan is not None:row['actor_reference']=ppo_plan['rows'][index]['run_dir']+'/initial-model.zip'
            rows.append(row)
    return rows


def load_plan(root,path,verify_source=True):
    root=Path(root).resolve();path=checked_path(root,path)
    plan=read(path)
    if plan['schema']!=1 or plan['algorithm'] not in ('ppo','mappo'):raise ValueError('Unknown matrix schema or algorithm')
    validate_config(plan['config'])
    if verify_source and source_hashes(root)!=plan['source_sha256']:raise ValueError('Matrix source changed; use its frozen source root')
    parent=None
    if plan['algorithm']=='mappo':
        reference=checked_path(root,plan['ppo_plan'])
        if reference==path:raise ValueError('A matrix cannot reference itself')
        if sha(reference)!=plan['ppo_plan_sha256']:raise ValueError('PPO reference plan changed')
        parent=read(reference)
        if parent['algorithm']!='ppo':raise ValueError('MAPPO must reference a PPO matrix')
        parent=load_plan(root,reference,verify_source=verify_source)
        if parent['config']!=plan['config']:raise ValueError('PPO and MAPPO configurations differ')
    if plan['rows']!=expected_rows(root,path,plan['algorithm'],plan['config'],parent):
        raise ValueError('Matrix must contain every support/seed pair once with its matching reference')
    return plan


def create_plan(root,path,config=None,ppo_plan_path=None):
    root=Path(root).resolve();path=checked_path(root,path)
    if path.exists():raise ValueError('Choose a fresh matrix path')
    parent=None;algorithm='ppo'
    if ppo_plan_path is not None:
        reference=checked_path(root,ppo_plan_path);parent=load_plan(root,reference)
        if parent['algorithm']!='ppo':raise ValueError('Expected a PPO parent matrix')
        algorithm='mappo';config=parent['config']
    validate_config(config)
    plan={'schema':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'algorithm':algorithm,
          'config':config,'source_sha256':source_hashes(root),
          'rows':expected_rows(root,path,algorithm,config,parent),
          'interpretation':'Retain all three training seeds and all four support settings; a best seed is not replication evidence.'}
    if parent is not None:plan.update(ppo_plan=reference.relative_to(root).as_posix(),ppo_plan_sha256=sha(reference))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(plan,stream,indent=2)
    return plan


def validate_reference(root,row,config,auditor=None):
    if row['algorithm']!='mappo':return None
    checkpoint=checked_path(root,row['actor_reference']);run=checkpoint.parent
    summary=read(run/'training_summary.json');actual=read(run/'config.json')
    if summary['status']!='complete':raise ValueError('PPO reference training did not complete its budget')
    for key,value in {**config,'algorithm':'ppo','seed':row['seed'],'guidance':row['guidance'],'filter':row['filter']}.items():
        if actual.get(key)!=value:raise ValueError('PPO reference configuration mismatch: '+key)
    manifest=read(run/'checkpoints.json')
    entries=[r for r in manifest if r['file']=='initial-model.zip']
    if len(entries)!=1 or any(entries[0][k]!=0 for k in ('live_transitions','counted_transitions','optimizer_steps')):
        raise ValueError('MAPPO reference must be the zero-experience initial actor')
    if sha(checkpoint)!=entries[0]['sha256']:raise ValueError('Initial PPO checkpoint integrity mismatch')
    if auditor is None:
        from atc_rl.audit import audit
        auditor=audit
    result=auditor(run)
    if result['status']!='complete':raise ValueError('PPO reference audit did not confirm completion')
    return result


def training_command(root,row,config,device,max_wall_seconds):
    command=[sys.executable,'-u','-m','atc_rl.train','--algorithm',row['algorithm'],
             '--seed',str(row['seed']),'--run-dir',str(checked_path(root,row['run_dir'])),
             '--device',device,'--checkpoint-live-steps','100000','--max-wall-seconds',str(max_wall_seconds)]
    for key,value in config.items():
        if key=='neutral_action_mean':
            if value:command.append('--neutral-action-mean')
        else:command.extend(['--'+key.replace('_','-'),str(value)])
    if row['guidance']:command.append('--guidance')
    if row['filter']:command.append('--filter')
    if row['algorithm']=='mappo':command.extend(['--actor-reference',str(checked_path(root,row['actor_reference']))])
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    create=commands.add_parser('create');create.add_argument('--plan',type=Path,required=True)
    create.add_argument('--ppo-plan',type=Path,help='Create matched MAPPO rows by inheriting this PPO matrix')
    create.add_argument('--workers',type=int);create.add_argument('--live-steps',type=int)
    create.add_argument('--action-reference',choices=['direct','goal_offset'])
    create.add_argument('--initial-action-std',type=float);create.add_argument('--neutral-action-mean',action='store_true',default=None)
    create.add_argument('--reward-scale',type=float);create.add_argument('--progress-scale',type=float)
    run=commands.add_parser('run');run.add_argument('--plan',type=Path,required=True);run.add_argument('--index',type=int,required=True)
    run.add_argument('--device',choices=['cpu','cuda'],default='cuda');run.add_argument('--max-wall-seconds',type=float,default=9900.)
    run.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.action=='create':
        if args.ppo_plan:
            if any(getattr(args,key,None) is not None for key in CONFIG_KEYS):
                parser.error('MAPPO inherits the PPO recipe; omit recipe overrides')
            plan=create_plan(root,args.plan,ppo_plan_path=args.ppo_plan)
        else:
            if any(getattr(args,key) is None for key in ('action_reference','initial_action_std','reward_scale')):
                parser.error('Select action-reference, initial-action-std and reward-scale explicitly for PPO')
            config=dict(workers=8,live_steps=1000000,rollout_steps=256,batch_size=1024,epochs=10,
                        neutral_action_mean=False,progress_scale=0.)
            config.update({key:getattr(args,key) for key in CONFIG_KEYS if getattr(args,key,None) is not None})
            plan=create_plan(root,args.plan,config)
        print(json.dumps({'plan':str(checked_path(root,args.plan)),'algorithm':plan['algorithm'],'array_rows':len(plan['rows']),'submitted_jobs':0}));return
    plan=load_plan(root,args.plan)
    if not 0<=args.index<len(plan['rows']):parser.error('Matrix index must be between 0 and 11')
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds<=0:parser.error('Use a finite positive wall limit')
    if 'SLURM_CPUS_PER_TASK' in os.environ and int(os.environ['SLURM_CPUS_PER_TASK'])<plan['config']['workers']+2:
        raise ValueError('Request CPUs for simulator workers plus learner and coordination')
    row=plan['rows'][args.index];directory=checked_path(root,row['run_dir'])
    command=training_command(root,row,plan['config'],args.device,args.max_wall_seconds)
    if args.dry_run:print(json.dumps({'command':command,'training_started':False,'mappo_reference_check_required_at_launch':row['algorithm']=='mappo'}));return
    if directory.exists():raise ValueError('Run directory exists; preserve it and create a fresh matrix for another attempt')
    reference_audit=validate_reference(root,row,plan['config'])
    receipt=checked_path(root,args.plan).parent/'launches'/f'row-{args.index:02d}.json';receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x',encoding='utf-8') as stream:json.dump({'command':command,'row':row,'plan_sha256':sha(checked_path(root,args.plan)),
        'reference_audit':reference_audit,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID')},stream,indent=2)
    subprocess.run(command,cwd=root,check=True)
    subprocess.run([sys.executable,'-m','atc_rl.audit','--run',str(directory),'--out',str(directory/'checkpoint-audit.json')],cwd=root,check=True)
    if read(directory/'training_summary.json')['status']!='complete':raise RuntimeError('Partial training retained; requested experience budget was not completed')
    print(json.dumps({'run':str(directory),'status':'complete','row':args.index}))


if __name__=='__main__':main()
