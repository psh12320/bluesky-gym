"""Execute one fixed training-seed replication; preserve every stage and outcome."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
for key,value in {'SDL_VIDEODRIVER':'dummy','PYGAME_HIDE_SUPPORT_PROMPT':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}.items():os.environ.setdefault(key,value)
parser=argparse.ArgumentParser()
parser.add_argument('--track',choices=['ma','sa'],required=True)
parser.add_argument('--seed',type=int,choices=[2902,2904],required=True)
parser.add_argument('--recover-training-audit',action='store_true')
args=parser.parse_args()
base=Path(__file__).resolve().parent
protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
out=base/f'{args.track}-seed{args.seed}'
if not args.recover_training_audit:
    out.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
state={'track':args.track,'training_seed':args.seed,'started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(base/'protocol.json'),'runner_sha256':sha(Path(__file__)),'stages':[]}
if args.recover_training_audit:
    previous=read(out/'state.json')
    assert previous['status']=='failed' and not (out/'candidate').exists()
    assert previous['protocol_sha256']==state['protocol_sha256']
    assert previous['track']==args.track and previous['training_seed']==args.seed
    assert len(previous['stages'])==1 and previous['stages'][0]['name']=='train' and previous['stages'][0]['exit_code']==0
    assert not (out/'state-before-audit-recovery.json').exists()
    shutil.copyfile(out/'state.json',out/'state-before-audit-recovery.json')
    state['stages']=previous['stages']
    state['audit_recovery']={'previous_error':previous['error'],'reason':'Normalize omitted requested critic warmup to zero; verify actual warmup and all remaining configuration fields. Reuse completed training and immutable callback, without another training run.','recovered_at_utc':datetime.now(timezone.utc).isoformat()}
def save(): (out/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
def stage(label,module,*arguments):
    command=[sys.executable,'-u','-m',module,*map(str,arguments)]
    record={'name':label,'command':command,'started_at_utc':datetime.now(timezone.utc).isoformat()}
    state['stages'].append(record);save()
    print(f'Starting {args.track} seed {args.seed}: {label}',flush=True)
    with (out/f'{label}.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        record['pid']=child.pid;save()
        for line in child.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=child.wait()
    record['exit_code']=code;record['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
    if code: raise RuntimeError(f'{label} exited {code}; keep evidence and do not skip this replica')
try:
    for track,original in protocol['primary_models'].items():
        assert sha(ROOT/original['model'])==original['model_sha256']
    for name,digest in protocol['primary_source_sha256'].items():
        if Path(name).suffix in {'.py','.toml','.slurm','.sh','.ps1','.yaml','.yml','.lock'}:
            assert sha(ROOT/name)==digest,name
    recipe=protocol['training']['recipes'][args.track]
    family='fast-reference' if args.track=='ma' else 'fast-reference-reach250'
    training=ROOT/f'runs/sac-ma-{family}-v1-25k-seed{args.seed}'
    original_training=ROOT/f'runs/sac-ma-{family}-v1-25k-seed2900'
    if args.recover_training_audit or (args.track=='ma' and args.seed==2902):
        assert training.exists()
        state['reused_existing_training']=True
    else:
        assert not training.exists()
        stage('train','atc.train','--env','ma','--algorithm','sac','--recipe',recipe,'--guard-traffic','--workers',1,'--steps',25000,'--seed',args.seed,'--device','cpu','--run-dir',training,'--checkpoint-every',25000,'--learning-starts',5000,'--gradient-steps',4,'--batch-size',256,'--buffer-size',100000,'--save-replay','--critic-warmup-updates',0,'--max-wall-seconds',0)
    summary=read(training/'training_summary.json')
    assert summary['timesteps']==25000 and summary['critic_updates']==8000
    assert summary['live_transitions']+summary['skipped_transitions']==25000
    checkpoint=training/'checkpoints/model_25000_steps.zip'
    with zipfile.ZipFile(checkpoint) as archive:
        data=json.loads(archive.read('data'))
    assert data['num_timesteps']==25000 and data['_n_updates']==7996
    config=read(training/'config.json'); original=read(original_training/'config.json')
    ignored={'seed','world_seeds','run_dir','max_wall_seconds'}
    assert config['actual_critic_warmup_updates']==original['actual_critic_warmup_updates']==0
    config['critic_warmup_updates']=config['critic_warmup_updates'] or 0
    original['critic_warmup_updates']=original['critic_warmup_updates'] or 0
    assert {k:v for k,v in config.items() if k not in ignored}=={k:v for k,v in original.items() if k not in ignored}
    assert config['world_seeds']==[args.seed] and config['recipe']==recipe
    import torch,io
    def tensors(path):
        with zipfile.ZipFile(path) as archive:
            return torch.load(io.BytesIO(archive.read('policy.pth')),map_location='cpu',weights_only=True)
    initial=tensors(training/'initial-model.zip'); initial_original=tensors(original_training/'initial-model.zip')
    assert initial.keys()==initial_original.keys()
    assert any(not torch.equal(initial[k],initial_original[k]) for k in initial if k.startswith('actor.latent_pi.'))
    assert torch.count_nonzero(initial['actor.mu.weight'])==0 and torch.count_nonzero(initial['actor.mu.bias'])==0
    del initial,initial_original
    state['training_audit']={'exact_25000_step_callback':True,'callback_optimizer_updates':7996,'final_optimizer_updates':8000,'matched_configuration_except':sorted(ignored),'initial_hidden_actor_weights_differ_from_primary':True,'initial_actor_mean_zero':True,'checkpoint_sha256':sha(checkpoint),'training_summary':summary}
    save()
    candidate=out/'candidate'
    if args.track=='sa':
        stage('transfer','atc.transfer_track','--model',checkpoint,'--out-dir',candidate)
    else:
        from atc.provenance import capture
        from atc.recipes import RECIPES
        from atc.submission import _validate_configuration
        candidate.mkdir();capture(candidate)
        shutil.copyfile(checkpoint,candidate/'model.zip')
        deployment=dict(config)
        deployment.update(recipe='public_route_choice_fast_residual_interval5',run_dir=str(candidate),source_model=str(checkpoint),source_model_sha256=sha(checkpoint),source_training_configuration=config,source_training_decision_interval_seconds=10,additional_training_transitions=0,initialization='Post-training decision-interval ablation; checkpoint bytes unchanged')
        deployment.update(RECIPES[deployment['recipe']].action_configuration())
        _validate_configuration('ma',deployment)
        (candidate/'config.json').write_text(json.dumps(deployment,indent=2),encoding='utf-8')
    assert sha(candidate/'model.zip')==sha(checkpoint)
    deployment=read(candidate/'config.json')
    assert deployment['recipe']==protocol['primary_models'][args.track]['recipe']
    state['deployed_model_sha256']=sha(candidate/'model.zip');save()
    for label,seed,episodes in [('development-20',2026,20),('replication-200',2027,200)]:
        stage(label,'atc.evaluate','--env',args.track,'--algorithm','sac','--recipe',deployment['recipe'],'--model',candidate/'model.zip','--episodes',episodes,'--seed',seed,'--out',out/label)
        from atc.compare import load_evaluation
        from atc.metrics import summarize
        meta,rows=load_evaluation(out/label)
        assert meta['model_sha256']==sha(candidate/'model.zip')
        expected=summarize(rows,episodes,10 if args.track=='ma' else 1)
        assert all(meta[k]==v for k,v in expected.items())
        state.setdefault('evaluations',{})[label]={'seed':seed,'episodes':episodes,'summary_recomputed':True,'csv_sha256':sha(out/(label+'.csv')),'summary':expected};save()
    reference=ROOT/('runs/heldout-2027-ma-interval5-v1/classical' if args.track=='ma' else 'runs/heldout-2027-sa-reach250-v1/classical')
    if reference.with_suffix('.csv').exists():
        stage('comparison','atc.compare',reference,out/'replication-200','--out',out/'comparison.json')
    for track,original in protocol['primary_models'].items():
        assert sha(ROOT/original['model'])==original['model_sha256']
    state['completed_at_utc']=datetime.now(timezone.utc).isoformat();state['status']='complete';save()
except BaseException as error:
    state['status']='failed';state['error']=repr(error);save();raise
