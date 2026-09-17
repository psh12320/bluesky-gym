"""Evaluate the registered untrained support controls sequentially."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,subprocess,sys

root=Path(__file__).resolve().parents[2]
parent=Path(__file__).resolve().parent
protocol=json.loads((parent/'protocol.json').read_text())
frozen=root/protocol['evaluation_source']
manifest=json.loads((frozen/'cluster-manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((frozen/name).read_bytes()).hexdigest()==digest,name
for cell in protocol['controls']:
    if cell['reused_evaluation']:continue
    model=Path(cell['checkpoint']);destination=Path(cell['evaluation'])
    assert not destination.exists(),str(destination)
    assert hashlib.sha256(model.read_bytes()).hexdigest()==protocol['source_initial_model_sha256']
    command=[sys.executable,'-m','atc_rl.evaluate','--model',str(model),
             '--episodes',str(protocol['evaluation_worlds']),'--seed',str(protocol['evaluation_seed']),
             '--out',str(destination)]
    with (parent/(cell['name']+'-launch.json')).open('x',encoding='utf-8') as stream:
        json.dump({'command':command,'cwd':str(frozen),'started_at_utc':datetime.now(timezone.utc).isoformat()},stream,indent=2)
    subprocess.run(command,cwd=frozen,check=True)
    summary=json.loads((destination/'summary.json').read_text())
    assert summary['episodes']==protocol['evaluation_worlds']
    assert hashlib.sha256((destination/'aircraft.csv').read_bytes()).hexdigest()==summary['csv_sha256']
print(json.dumps({'completed_support_cells':3,'additional_training_transitions':0}))
