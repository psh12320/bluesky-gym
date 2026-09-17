"""Run the frozen SA candidate through the original 1000-scenario harness."""
import ast
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

for name,value in {'SDL_VIDEODRIVER':'dummy','PYGAME_HIDE_SUPPORT_PROMPT':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}.items():
    os.environ.setdefault(name,value)
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
assert not (OUT/'protocol.json').exists(), 'A run already exists; do not overwrite official evidence'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
heldout=ROOT/'runs/heldout-2027-sa-reach250-v1'
freeze=json.loads((heldout/'protocol.json').read_text())
audit=json.loads((heldout/'completion-audit.json').read_text())
assert audit['all_summaries_recomputed'] and audit['paired_comparison_recomputed']
assert audit['evaluations']['learned']['summary']['metrics']['waypoint_reached']['mean']==1.0
model=ROOT/freeze['model']
assert sha(model)==freeze['model_sha256']
assert json.loads((model.parent/'config.json').read_text())==freeze['configuration']
for name,digest in freeze['source_sha256'].items():
    if Path(name).suffix in {'.py','.toml','.slurm','.sh','.ps1','.yaml','.yml','.lock'}:
        assert sha(ROOT/name)==digest, name
harness=ROOT/'scripts/evaluate_competition.py'
original=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','show','HEAD:scripts/evaluate_competition.py'],text=True,encoding='utf-8')
def outside_hooks(source):
    tree=ast.parse(source)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name in {'make_env','load_policy'})]
    return ast.dump(tree,include_attributes=False)
assert outside_hooks(original)==outside_hooks(harness.read_text(encoding='utf-8'))
from scripts import evaluate_competition as verification
assert verification.SEED==42 and verification.N_EPISODES==1000 and verification.N_AGENTS_MA==10
from atc.provenance import capture
snapshot=capture(OUT/'source')
(OUT/'candidate').mkdir()
shutil.copyfile(model,OUT/'candidate/model.zip')
shutil.copyfile(model.parent/'config.json',OUT/'candidate/config.json')
assert sha(OUT/'candidate/model.zip')==freeze['model_sha256']
command=['scripts.evaluate_competition','--env','sa','--episodes','1000','--model',str(OUT/'candidate/model.zip'),'--out',str(OUT/'metrics.csv')]
protocol={'started_at_utc':datetime.now(timezone.utc).isoformat(),'phase':'Local run of the original official scoring protocol; not a judge-verified submission','env':'sa','seed':42,'episodes':1000,'seed_once_then_continue':True,'model_sha256':freeze['model_sha256'],'configuration':freeze['configuration'],'heldout_protocol_sha256':sha(heldout/'protocol.json'),'heldout_audit_sha256':sha(heldout/'completion-audit.json'),'harness_sha256':sha(harness),'outside_two_allowed_hooks_ast_matches_head':True,'source_commit':snapshot['git_commit'],'source_sha256':snapshot['source_sha256'],'command':[sys.executable,'-u','-m',*command],'decision':'Promote the frozen SA feasibility candidate unchanged after 200/200 held-out arrivals versus 194/200 classical; retain flight and safety tradeoffs. No model selection or tuning on seed 42.','ma_candidate_already_frozen':True,'no_tuning_or_checkpoint_selection_on_official_outcomes':True}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
print(json.dumps({k:protocol[k] for k in ('started_at_utc','env','seed','episodes','model_sha256','outside_two_allowed_hooks_ast_matches_head')},indent=2),flush=True)
class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,text):
        for stream in self.streams: stream.write(text); stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()
started=time.perf_counter()
with (OUT/'harness.log').open('x',encoding='utf-8') as log:
    with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        sys.argv=command
        runpy.run_module('scripts.evaluate_competition',run_name='__main__')
wall=time.perf_counter()-started
from atc.metrics import METRICS,summarize
with (OUT/'metrics.csv').open(newline='',encoding='utf-8') as stream:
    raw=list(csv.DictReader(stream))
assert [int(r['episode_index']) for r in raw]==list(range(1000))
records=[{'episode':i,'agent':'KL001',**{k:float(r[k]) for k in METRICS}} for i,r in enumerate(raw)]
summary=summarize(records,1000,1)
assert sha(harness)==protocol['harness_sha256']
assert sha(OUT/'candidate/model.zip')==protocol['model_sha256']
summary.update({'completed_at_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':wall,'env':'sa','seed':42,'official_protocol':True,'judge_verified':False,'model_sha256':protocol['model_sha256'],'csv_sha256':sha(OUT/'metrics.csv'),'protocol_sha256':sha(OUT/'protocol.json'),'harness_source_unchanged_during_run':True})
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
