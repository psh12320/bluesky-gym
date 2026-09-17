"""Verify the arrival-reward ablation against exact saved trajectories."""
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import numpy as np
import torch
from stable_baselines3 import SAC
import bluesky as bs
from atc.compare import load_evaluation
from atc.envs import make_env
from atc.inference import predict_actions
from atc.metrics import METRICS, summarize
from atc.provenance import capture
from core.scenario import Scenario, Obstacle, AgentSpec, Route
from core.tools import kwikqdrdist

out = Path(__file__).resolve().parent
scenario_path = ROOT / 'runs/diagnose-fast-reference-v1-sa-episode28/scenario.json'
raw = json.loads(scenario_path.read_text())
scenario = Scenario(center=raw['center'], sector=raw['sector'],
                    obstacles=[Obstacle(**x) for x in raw['obstacles']],
                    agents=[AgentSpec(**x) for x in raw['agents']],
                    intruder_routes=[Route(**x) for x in raw['intruder_routes']])
assert json.loads(json.dumps(asdict(scenario))) == raw
model_path = ROOT / 'runs/fast-reference-v1-ma25k-sa/model.zip'
torch.set_num_threads(1)
model = SAC.load(model_path, device='cpu', buffer_size=1)
_, reference = load_evaluation('runs/fast-reference-v1-ma25k-sa/validation-200')
_, classical = load_evaluation('runs/goal-route-choice-v1-fast-joint-sa-200')
expected = {name: next(r for r in rows if r['episode'] == 28) for name, rows in [('learned10', reference), ('zero10', classical)]}
capture(out / 'provenance')
report = dict(model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
              scenario_sha256=hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              scope='Arrival-reward verification on a selected scenario; no policy training or population claim',
              reach_reward=250.0, original_reach_reward=37.41716380974046,
              zero_control='Exactly zero residual output reproduces the fast classical route follower',
              arms={})
for name, interval, zero in [('learned10', 10, False), ('zero10', 10, True)]:
    env = make_env('sa', 'public_route_choice_fast_residual_reach250_sa_transfer', guard_traffic=True)
    trace = []
    started = time.perf_counter()
    try:
        env.unwrapped._fixed_scenario = scenario
        obs, _ = env.reset(seed=2026)
        world = env.unwrapped
        world.action_frequency = interval
        assert world.sim_dt == 1 and world.d_heading == 45 and world.speed_action.d_speed == 20/3
        assert world.intrusion_distance == 5 and world.distance_margin == 5
        rng = repr(world._np_random.bit_generator.state)
        while True:
            action = np.zeros(2, dtype=np.float32) if zero else predict_actions(model, obs)
            i = bs.traf.id2idx(world.agent)
            position = world._xy((bs.traf.lat[i], bs.traf.lon[i]))
            bearing, target_distance = world._route_reference(world.agent)
            target = position + target_distance * np.array([np.sin(np.deg2rad(bearing)), np.cos(np.deg2rad(bearing))])
            _, distance = kwikqdrdist(bs.traf.lat[i], bs.traf.lon[i], world.goal_lat, world.goal_lon)
            row = dict(time=float(world.metrics[world.agent]['flight_time']), east_km=float(position[0]), north_km=float(position[1]),
                       goal_distance_km=float(distance*1.852), heading_deg=float(bs.traf.hdg[i]), cas_mps=float(bs.traf.cas[i]),
                       target_east_km=float(target[0]), target_north_km=float(target[1]), target_distance_km=target_distance,
                       clearance_km=float(world._planners[world.agent].clearance), residual_heading=float(action[0]), residual_speed=float(action[1]))
            obs, _, terminated, truncated, info = env.step(action)
            row.update(world.last_projection.get(world.agent, {}))
            trace.append(row)
            if terminated or truncated:
                record = dict(episode=0, agent=world.agent, **info)
                break
        assert repr(world._np_random.bit_generator.state) == rng
        result = dict(summary=summarize([record], 1, 1), record=record, decisions=len(trace),
                      decision_interval_seconds=interval, wall_seconds=time.perf_counter()-started,
                      scenario_rng_unchanged=True, route_statistics=dict(world.route_input_statistics), traffic_statistics=dict(world.traffic_projection_statistics))
        if name in expected:
            reward_delta = (250.0 - 37.41716380974046) * record['waypoint_reached']
            differences = {k: abs(record[k] - expected[name][k] - (reward_delta if k == 'total_reward' else 0.0)) for k in METRICS}
            assert all(value <= (1e-10 if key == 'total_reward' else 0.0) for key, value in differences.items()), differences
            result.update(all_eight_objective_metrics_exact=True, expected_total_reward_delta=reward_delta,
                          reward_change_matches_expected=True, maximum_absolute_expected_differences=differences)
        fields = list(dict.fromkeys(k for row in trace for k in row))
        with (out/(name+'-trajectory.csv')).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(trace)
        result['trajectory_sha256'] = hashlib.sha256((out/(name+'-trajectory.csv')).read_bytes()).hexdigest()
        report['arms'][name] = result
    finally:
        env.close()
    (out/'targeted-results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'arm':name, 'result':result}), flush=True)
