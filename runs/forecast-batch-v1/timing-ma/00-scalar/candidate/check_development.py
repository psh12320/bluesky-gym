from pathlib import Path
import argparse,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
os.chdir(SOURCE)
sys.path.insert(0,str(SOURCE))
from atc import deployment
import atc,core,bluesky_gym,bluesky_zoo
for module in (atc,core,bluesky_gym,bluesky_zoo):
    assert SOURCE in Path(module.__file__).resolve().parents, module.__name__
from scripts import evaluate_competition as harness
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--env',choices=['sa','ma'],required=True)
args=parser.parse_args()
expected=json.loads((ROOT/f'evidence/integration-{args.env}.json').read_text())
output=ROOT/f'checks/{args.env}.json'
assert not output.exists(), 'Use a fresh extraction to repeat this check'
model=ROOT/f'models/{args.env}/model.zip'
assert hashlib.sha256(model.read_bytes()).hexdigest()==expected['model_sha256']
harness.make_env=deployment.make_env
harness.load_policy=deployment.load_policy
harness.SEED=2026
act=harness.load_policy(args.env,model)
records=harness.run_single_agent(2,act) if args.env=='sa' else harness.run_multi_agent(2,act,10)
assert len(records)==len(expected['actual_records'])
delta={k:float(np.max(np.abs(np.array([r[k] for r in records])-np.array([r[k] for r in expected['actual_records']])))) for k in harness.METRIC_KEYS}
assert all(v<=(1e-5 if k=='total_reward' else 0) for k,v in delta.items())
output.parent.mkdir(exist_ok=True)
result={'env':args.env,'seed':2026,'episodes':2,'model_sha256':expected['model_sha256'],'extracted_source_used':True,'source_directory':str(SOURCE),'all_metrics_match':True,'metric_max_absolute_difference':delta,'dependency_environment':'Existing interpreter; dependencies were not reinstalled by this check'}
output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
