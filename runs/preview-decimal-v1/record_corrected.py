"""Record the first five development scenarios with explicit corrected deployment."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import shutil,sys
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
from atc import deployment,record
parser=argparse.ArgumentParser()
parser.add_argument('--env',choices=['sa','ma'],required=True)
args=parser.parse_args()
base=Path(__file__).resolve().parent
out=base/args.env
assert not out.exists()
model=ROOT/f'runs/decimal-transport-candidates-v1/{args.env}/model.zip'
manifest_path=model.parent/'deployment.json'
manifest=json.loads(manifest_path.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(model)==manifest['model_sha256']
reference=ROOT/('runs/fast-reference-reach250-v1-ma25k-sa/validation-200' if args.env=='sa'
                else 'runs/fast-reference-interval5-v1-ma25k-ma/validation-20')
protocol={'started_at_utc':datetime.now(timezone.utc).isoformat(),'env':args.env,
          'seed':2026,'episodes':list(range(5)),'heading_transport':manifest['heading_transport'],
          'model_sha256':sha(model),'deployment_manifest_sha256':sha(manifest_path),
          'runner_sha256':sha(Path(__file__)),'reference':str(reference),
          'purpose':'Corrected deployment preview checked against the original first five development scenarios; not a population or official score'}
protocol_path=base/f'{args.env}-protocol.json'
assert not protocol_path.exists()
protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
record.submission=deployment
previous_argv=sys.argv
error=None
try:
    sys.argv=['atc.record','--reference',str(reference),'--model',str(model),
              '--episodes','0','1','2','3','4','--out',str(out)]
    record.main()
except BaseException as exc:
    error=repr(exc)
    raise
finally:
    sys.argv=previous_argv
    if (out/'recording.json').exists():
        metadata=json.loads((out/'recording.json').read_text())
        metadata.update(heading_transport=manifest['heading_transport'],
                        deployment_manifest_sha256=sha(manifest_path),
                        recording_protocol_sha256=sha(protocol_path),
                        recording_purpose=protocol['purpose'],
                        recording_complete=error is None and len(metadata['episodes'])==5,
                        recording_error=error)
        (out/'recording.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        shutil.copyfile(Path(__file__),out/'recording-wrapper.py')
        shutil.copyfile(protocol_path,out/'recording-protocol.json')
    state={'status':'complete' if error is None else 'failed',
           'finished_at_utc':datetime.now(timezone.utc).isoformat(),'error':error,
           'protocol_sha256':sha(protocol_path)}
    (base/f'{args.env}-state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
