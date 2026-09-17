"""Check versioned deployment hooks against frozen development-prefix metrics."""
import argparse,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from atc import deployment
from atc.compare import load_evaluation
from atc.provenance import capture
from scripts import evaluate_competition as harness
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--env',choices=['sa','ma'],required=True)
args=parser.parse_args()
out=Path(__file__).resolve().parent
model=out/args.env/'model.zip'
manifest=out/args.env/'deployment.json'
reference=ROOT/('runs/fast-reference-reach250-v1-ma25k-sa/validation-200' if args.env=='sa' else 'runs/fast-reference-interval5-v1-ma25k-ma/validation-200')
meta,expected=load_evaluation(reference)
expected=[r for r in expected if r['episode']<2]
assert meta['model_sha256']==hashlib.sha256(model.read_bytes()).hexdigest()
assert not (out/f'integration-{args.env}.json').exists()
capture(out/f'integration-{args.env}-source')
harness.make_env=deployment.make_env
harness.load_policy=deployment.load_policy
harness.SEED=2026
act=harness.load_policy(args.env,model)
actual=harness.run_single_agent(2,act) if args.env=='sa' else harness.run_multi_agent(2,act,10)
assert len(actual)==len(expected)
differences={k:float(np.max(np.abs(np.asarray([r[k] for r in actual])-np.asarray([r[k] for r in expected])))) for k in harness.METRIC_KEYS}
assert all(v <= (1e-5 if k=='total_reward' else 0) for k,v in differences.items())
result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'env':args.env,'episodes':2,'agent_episodes':len(actual),'seed':2026,'official_protocol':False,'scope':'Original rollout functions with the corrected deployment integration hooks; development prefix only','model_sha256':meta['model_sha256'],'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'harness_sha256':hashlib.sha256((ROOT/'scripts/evaluate_competition.py').read_bytes()).hexdigest(),'heading_transport_revision':1,'reference':str(reference),'metric_max_absolute_differences':differences,'matches_reference':True,'actual_records':[{k:float(r[k]) for k in harness.METRIC_KEYS} for r in actual]}
(out/f'integration-{args.env}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='actual_records'},indent=2))
