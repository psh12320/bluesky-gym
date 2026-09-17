"""Matched fixed-scenario probe of a 20-knot speed-command increment."""
import csv
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
from atc.envs import make_env
from atc.metrics import METRICS
from atc.provenance import capture
from core.scenario import Scenario,Obstacle,AgentSpec,Route
import bluesky as bs
from core.tools import kwikqdrdist

torch.set_num_threads(1)
root=Path(__file__).resolve().parent
output=root/'targeted-results.json'
if output.exists(): raise FileExistsError(output)
source=ROOT/'runs/candidate-sa-route-residual-v1-ma25k/model.zip'
candidate=ROOT/'runs/route-choice-v1-ma25k-sa/model.zip'
model_hash=hashlib.sha256(source.read_bytes()).hexdigest()
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==model_hash
model=SAC.load(source,device='cpu',buffer_size=1)
capture(root/'provenance')
report=dict(model_sha256=model_hash, additional_training_transitions=0,
    analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Selected development failures, no training or population claim', speed_increment_knots=[20/3,20.0], unchanged='Routing, policy weights, heading mapping, 10-second action interval, original simulator and scoring', cases={})

def rollout(recipe,scenario,path,speed_increment):
    env=make_env('sa',recipe,guard_traffic=True)
    rows=[]
    started=time.perf_counter()
    try:
        env.unwrapped._fixed_scenario=scenario
        obs,_=env.reset(seed=2026)
        world=env.unwrapped
        world.d_speed=speed_increment
        world.speed_action.d_speed=speed_increment
        assert world.action_frequency==10 and world.d_heading==45
        done=False
        while not done:
            idx=bs.traf.id2idx(world.agent)
            _,distance=kwikqdrdist(bs.traf.lat[idx],bs.traf.lon[idx],world.goal_lat,world.goal_lon)
            rows.append(dict(time=world.metrics[world.agent]['flight_time'],
                lat=float(bs.traf.lat[idx]),lon=float(bs.traf.lon[idx]),
                cas_mps=float(bs.traf.cas[idx]),goal_distance_km=float(distance*1.852),
                route_clearance_km=float(world._planners[world.agent].clearance)))
            action=model.predict(obs,deterministic=True)[0]
            obs,_,terminated,truncated,info=env.step(action)
            done=terminated or truncated
        result=dict(metrics={k:float(info[k]) for k in METRICS},
            wall_seconds=time.perf_counter()-started,
            route_statistics=dict(world.route_input_statistics),
            traffic_statistics=dict(world.traffic_projection_statistics))
    finally:
        env.close()
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result

reference_results=json.loads((ROOT/'runs/route-choice-v1-audit/targeted-results.json').read_text())
for episode in (86,181,96,179):
    reference=ROOT/f'runs/diagnose-residual-v1-sa-episode{episode}'
    raw=json.loads((reference/'scenario.json').read_text())
    scenario=Scenario(center=raw['center'],sector=raw['sector'],
        obstacles=[Obstacle(**o) for o in raw['obstacles']],
        agents=[AgentSpec(**a) for a in raw['agents']],
        intruder_routes=[Route(**r) for r in raw['intruder_routes']])
    assert asdict(scenario)==raw
    original=rollout('public_route_choice_residual_sa_transfer',scenario,root/f'sa{episode}-original.csv',20/3)
    expected=reference_results['cases'][str(episode)]['route_choice']['metrics']
    differences={k:abs(original['metrics'][k]-expected[k]) for k in METRICS}
    report['cases'][str(episode)]=dict(scenario_sha256=hashlib.sha256((reference/'scenario.json').read_bytes()).hexdigest(),
        original=original,original_reference_differences=differences)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert not any(differences.values()),(episode,differences)
    print(json.dumps(dict(episode=episode,original_matches_reference=True)),flush=True)
    revised=rollout('public_route_choice_residual_sa_transfer',scenario,root/f'sa{episode}-speed20.csv',20.0)
    report['cases'][str(episode)]['speed20']=revised
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(episode=episode,original=original['metrics'],speed20=revised['metrics'])),flush=True)
