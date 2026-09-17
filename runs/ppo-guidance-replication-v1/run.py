from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,hashlib
root=Path(__file__).resolve().parents[2];parent=Path(__file__).resolve().parent
protocol=json.loads((parent/'protocol.json').read_text())
frozen=root/'runs/onpolicy-learning-source-v2-verify'
manifest=json.loads((frozen/'cluster-manifest.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256((frozen/name).read_bytes()).hexdigest()==digest,name
for seed in protocol['additional_seeds']:
    run=parent/f'seed-{seed}'/'train';assert not run.exists()
    command=[sys.executable,'-m','atc_rl.train','--algorithm','ppo','--workers','2','--live-steps','100000',
        '--rollout-steps','256','--batch-size','1024','--epochs','10','--seed',str(seed),'--device','cpu',
        '--initial-action-std','.05','--neutral-action-mean','--action-reference','goal_offset','--guidance',
        '--reward-scale','.01','--checkpoint-live-steps','50000','--max-wall-seconds','3600','--run-dir',str(run)]
    run.parent.mkdir(exist_ok=False)
    with (run.parent/'launch.json').open('x',encoding='utf-8') as stream:
        json.dump({'command':command,'cwd':str(frozen),'started_at_utc':datetime.now(timezone.utc).isoformat()},stream,indent=2)
    subprocess.run(command,cwd=frozen,check=True)
    subprocess.run([sys.executable,'-m','atc_rl.audit','--run',str(run),'--out',str(run.parent/'final-checkpoint-audit.json')],cwd=frozen,check=True)
    summary=json.loads((run/'training_summary.json').read_text())
    if summary['status']!='complete':raise RuntimeError('Replication stopped before its budget; preserve it and inspect before proceeding.')
print(json.dumps({'additional_seed_training_complete':protocol['additional_seeds'],'performance_evaluation':'pending'}))
