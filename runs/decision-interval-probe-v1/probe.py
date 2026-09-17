"""Matched single-case test of five-second versus ten-second control decisions."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import torch
from stable_baselines3 import SAC
from atc.compare import load_evaluation
from atc.envs import make_env
from atc.inference import predict_actions
from atc.metrics import METRICS,summarize
from atc.provenance import capture
from core.scenario import Scenario,Obstacle,AgentSpec,Route

output=Path(__file__).resolve().parent
model_path=ROOT/'runs/sac-ma-fast-reference-v1-25k-seed2900/checkpoints/model_25000_steps.zip'
scenario_path=ROOT/'runs/diagnose-fast-reference-v1-ma-episode19/scenario.json'
raw=json.loads(scenario_path.read_text())
scenario=Scenario(center=raw['center'],sector=raw['sector'],
    obstacles=[Obstacle(**x) for x in raw['obstacles']],
    agents=[AgentSpec(**x) for x in raw['agents']],
    intruder_routes=[Route(**x) for x in raw['intruder_routes']])
assert json.loads(json.dumps(asdict(scenario)))==raw
torch.set_num_threads(1)
model=SAC.load(model_path,device='cpu',buffer_size=1)
_,reference=load_evaluation(ROOT/'runs/sac-ma-fast-reference-v1-25k-seed2900/validation-20')
expected={r['agent']:r for r in reference if r['episode']==19}
capture(output/'provenance')
report=dict(model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    scenario_sha256=hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Selected MA development failure; no population or additional training claim',
    unchanged='Policy weights, nominal per-command ranges, route planning, joint filter, original simulator and one-second scoring',
    caveat='More frequent decisions also permit more frequent heading and speed changes; this is not an isolated observation-latency change.',
    arms={})
for interval in (10,5):
    env=make_env('ma','public_route_choice_fast_residual',guard_traffic=True)
    records=[]
    decisions=0
    started=time.perf_counter()
    try:
        env.unwrapped._fixed_scenario=scenario
        obs,_=env.reset(seed=2026)
        world=env.unwrapped
        world.action_frequency=interval
        assert world.sim_dt==1 and world.d_heading==45 and world.speed_action.d_speed==20/3
        assert world.intrusion_distance==5 and world.distance_margin==5
        rng=repr(world._np_random.bit_generator.state)
        while env.agents:
            ids=list(env.agents)
            commands=predict_actions(model,np.stack([obs[a] for a in ids]),batched=False)
            obs,_,terminated,truncated,infos=env.step(dict(zip(ids,commands)))
            decisions+=1
            for agent in ids:
                if terminated[agent] or truncated[agent]:
                    records.append(dict(episode=0,agent=agent,**infos[agent]))
        assert repr(world._np_random.bit_generator.state)==rng
        result=dict(summary=summarize(records,1,10),records=records,world_decisions=decisions,
            wall_seconds=time.perf_counter()-started,scoring_interval_seconds=world.sim_dt,
            traffic_statistics=dict(world.traffic_projection_statistics))
        if interval==10:
            actual={r['agent']:r for r in records}
            differences={k:max(abs(actual[a][k]-expected[a][k]) for a in expected) for k in METRICS}
            assert not any(differences.values()),differences
            result['all_nine_reference_metrics_exact']=True
            result['maximum_absolute_metric_differences']=differences
        report['arms'][str(interval)]=result
    finally:
        env.close()
    (output/'targeted-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(interval_seconds=interval,summary=result['summary'],world_decisions=decisions)),flush=True)
