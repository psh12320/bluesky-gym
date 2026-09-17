"""Inspect reset-time scored events on the existing development sequence."""
from pathlib import Path
from dataclasses import asdict
import csv, hashlib, json, os, sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault('OMP_NUM_THREADS','1')
from atc.envs import make_env
from atc.compare import load_evaluation
from atc.metrics import METRICS,SAFETY
from atc.provenance import capture
import bluesky as bs
root=Path(__file__).resolve().parent;capture(root/'provenance')
_,finals=load_evaluation('runs/candidate-sa-route-residual-v1-ma25k/validation-200')
finals={r['episode']:r for r in finals}
rows=[];checks={}
env=make_env('sa','baseline')
try:
    for episode in range(200):
        _,info=env.reset(seed=2026 if episode==0 else None)
        world=env.unwrapped
        rows.append(dict(episode=episode,**{k:float(info[k]) for k in METRICS}))
        assert finals[episode]['intrusion_events']>=info['intrusion_events']
        if episode in (86,96,179,181):
            path=ROOT/f'runs/diagnose-residual-v1-sa-episode{episode}/scenario.json'
            same=json.loads(json.dumps(asdict(world.scenario)))==json.loads(path.read_text())
            assert same,episode
            checks[str(episode)]=dict(exact_saved_scenario_match=same,scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        bs.stack.process()
finally:env.close()
with (root/'sa-reset-metrics.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report=dict(seed=2026,episodes=200,scenario_checks=checks,
    scope='Reset-time scored events only; no official metrics removed or adjusted',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
for count in (20,200):
    selected=rows[:count]
    initial=[r['episode'] for r in selected if r['intrusion_events']>0]
    initial_any=[r['episode'] for r in selected if any(r[k]>0 for k in SAFETY)]
    later=[r['episode'] for r in selected if r['episode'] not in initial and finals[r['episode']]['intrusion_events']>0]
    report[str(count)]=dict(initial_intrusion_scenarios=initial,
        initial_intrusion_count=len(initial),initial_scored_safety_scenarios=initial_any,
        maximum_possible_clean_completion_rate=1-len(initial_any)/count,
        retained_policy_conflict_scenarios_without_initial_intrusion=later,
        initial_metric_totals={k:sum(r[k] for r in selected) for k in SAFETY})
(root/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
