"""Package the corrected deployment and the evidence needed to inspect its integration."""
from pathlib import Path
import hashlib,json,sys,zipfile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from atc.provenance import capture
OUT=Path(__file__).resolve().parent
DEST=ROOT/'output/candidates/decimal-heading-v1.zip'
assert not DEST.exists()
snapshot=capture(OUT/'package-source')
files={}
with zipfile.ZipFile(OUT/'package-source/source.zip') as archive:
    assert archive.testzip() is None
    for name in archive.namelist():files['source/'+name]=archive.read(name)
for kind in ('ma','sa'):
    check=json.loads((OUT/f'integration-{kind}.json').read_text())
    assert check['matches_reference'] and check['heading_transport_revision']==1
    assert all(v==0 for v in check['metric_max_absolute_differences'].values())
    for name in ('model.zip','deployment.json'):
        files[f'models/{kind}/{name}']=(OUT/kind/name).read_bytes()
    files[f'evidence/integration-{kind}.json']=(OUT/f'integration-{kind}.json').read_bytes()
for label,path in {
    'original-command-diagnostic.json':'runs/heading-parser-ma47-v1/diagnostic.json',
    'corrected-command-diagnostic.json':'runs/heading-parser-ma47-decimal-v1/diagnostic.json',
    'command-correction-audit.json':'runs/heading-parser-ma47-decimal-v1/completion-audit.json',
    'source-isolation-audit.json':'runs/heading-parser-ma47-decimal-v1/source-isolation-audit.json',
    'runtime-bindings.json':'runs/frozen-runtime-compliance-v1/audit.json',
    'runtime-clock-and-resolution.json':'runs/frozen-runtime-compliance-v1/simulation-audit-canonical.json',
}.items():files['evidence/'+label]=(ROOT/path).read_bytes()
files['evidence/deployment-protocol.json']=(OUT/'protocol.json').read_bytes()
files['evidence/source-provenance.json']=(OUT/'package-source/provenance.json').read_bytes()
files['evidence/runtime-packages.txt']=(OUT/'package-source/packages.txt').read_bytes()
files['README.md']='''# Corrected heading deployment: candidate v1

This is an inspectable deployment candidate, not a completed competition submission.
It contains source, two selected trained models, versioned deployment manifests and
specific integration evidence. Full corrected evaluations are still pending.

The learned weights are unchanged. Heading transport expands scientific notation
to ordinary decimal because the installed simulator rejects the former. The original
and corrected known-case diagnostics are included, including the small reward change.
The source retains the original simulator/scoring files and the permitted integration
hooks. Historical experiment documents describe results beyond the evidence packaged here.

## Verify and inspect

From the extracted directory, run `python verify_contents.py`. This checks every
listed file against the package manifest. Model hashes and transport settings are in
`models/sa/deployment.json` and `models/ma/deployment.json`.

With the recorded Python dependencies available, run:

```sh
python check_development.py --env sa
python check_development.py --env ma
```

These checks import the extracted source, run two development scenarios per track,
and compare every metric with the included reference. They do not use seed 42.
Use Python 3.12. The package's runtime-packages.txt records the actual Windows CPU
environment, including a local editable-install path; it is evidence, not a portable
cross-platform lockfile. Dependency reinstallation has not been validated by this
small source/model extraction check.

For an installable checkout, clone the upstream competition branch, check out base
commit 00930013219af4c17e509c3efc84a8becd0f3546, and overlay `source/` into that fresh
checkout. The upstream build obtains its version from Git. Install that checkout in
a compatible environment and retain the package's model/manifest directories.

## Full scoring entry

From `source/` (or the overlaid checkout), with the model path adjusted as needed:

```sh
python -m atc.deployment --env sa --model ../models/sa/model.zip --out ../results/sa.csv
python -m atc.deployment --env ma --model ../models/ma/model.zip --out ../results/ma.csv
```

This uses the original 1000-scenario, seed-42 scoring loops and replaces only the two
allowed integration hooks. Use a new output file and retain the exact source,
manifest, installed versions and console output. These commands run local scoring;
they do not submit anything or establish a judge-verified ranking. The legacy loader
cannot load these manifest-only packages, which prevents silent omission of the fix.

The development checks and diagnostic replay have narrower scope than a full
validation. The source and model package remains provisional until the corrected
full evaluations and final report are complete.
'''.encode()
files['verify_contents.py']='''from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():
    path=(ROOT/name).resolve()
    assert ROOT in path.parents, name
    assert path.is_file(), name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, name
print(f"Verified {len(manifest['files'])} packaged files")
'''.encode()
files['check_development.py']='''from pathlib import Path
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
'''.encode()
files['package_source.py']=Path(__file__).read_bytes()
manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Source and unchanged weights for the decimal-heading deployment; full corrected scoring is pending','source_commit':snapshot['git_commit'],'files':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}}
files['manifest.json']=json.dumps(manifest,indent=2).encode()
DEST.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(DEST,'x',zipfile.ZIP_DEFLATED) as archive:
    for name,data in sorted(files.items()):archive.writestr(name,data)
with zipfile.ZipFile(DEST) as archive:
    assert archive.testzip() is None
    assert set(archive.namelist())==set(files)
    assert all(hashlib.sha256(archive.read(n)).hexdigest()==h for n,h in manifest['files'].items())
audit={'archive':str(DEST),'files':len(files),'bytes':DEST.stat().st_size,'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest(),'all_file_hashes_verified':True,'source_files':len(snapshot['source_sha256']),'model_manifests_verified':2,'extracted_runtime_check':'pending'}
(OUT/'package-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
