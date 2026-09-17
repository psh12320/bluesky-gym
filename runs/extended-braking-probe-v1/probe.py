"""Fixed-scenario experiment: add emergency braking only after joint infeasibility."""
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
from stable_baselines3 import SAC
from atc.compare import load_evaluation
from atc.envs import make_env
from atc.inference import predict_actions
from atc.metrics import METRICS, summarize
from atc.provenance import capture
import atc.traffic_projection as projection
from core.scenario import Scenario, Obstacle, AgentSpec, Route

OUTPUT = Path(__file__).resolve().parent
MODEL = ROOT / 'runs/sac-ma-fast-reference-v1-25k-seed2900/checkpoints/model_25000_steps.zip'
SCENARIO = ROOT / 'runs/diagnose-fast-reference-v1-ma-episode19/scenario.json'
raw = json.loads(SCENARIO.read_text())
scenario = Scenario(center=raw['center'], sector=raw['sector'],
    obstacles=[Obstacle(**value) for value in raw['obstacles']],
    agents=[AgentSpec(**value) for value in raw['agents']],
    intruder_routes=[Route(**value) for value in raw['intruder_routes']])
assert json.loads(json.dumps(asdict(scenario))) == raw
torch.set_num_threads(1)
model = SAC.load(MODEL, device='cpu', buffer_size=1)
original_choose = projection.choose_command
source = inspect.getsource(original_choose)
needle = 'speeds = np.unique(np.r_[nominal[1], -1, 0, 1])'
assert source.count(needle) == 1
extended_source = source.replace(needle, 'speeds = np.unique(np.r_[nominal[1], -3, -1, 0, 1])')
(OUTPUT / 'extended_choose_command.py').write_text(extended_source, encoding='utf-8')
namespace = dict(vars(projection))
exec(compile(extended_source, str(OUTPUT / 'extended_choose_command.py'), 'exec'), namespace)
extended_choose = namespace['choose_command']
capture(OUTPUT / 'provenance')
report = dict(model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
    scenario_sha256=hashlib.sha256(SCENARIO.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    original_choose_sha256=hashlib.sha256(source.encode()).hexdigest(),
    extended_choose_sha256=hashlib.sha256(extended_source.encode()).hexdigest(),
    scope='One selected development failure; no population or training claim',
    nominal_speed_increment_knots=20/3, emergency_speed_command=-3,
    emergency_speed_decrement_knots=20.0, additional_training_transitions=0,
    rule='Use the original result whenever it has a joint solution. Otherwise assess the same candidates plus normalized speed -3, through the original forecast and SpeedAction executor.',
    arms={})
_, reference = load_evaluation(ROOT / 'runs/sac-ma-fast-reference-v1-25k-seed2900/validation-20')
expected = {r['agent']: r for r in reference if r['episode'] == 19}

for arm in ('original', 'extended_braking'):
    counters = dict(joint_infeasibility_calls=0, extra_braking_selections=0)
    def emergency_choose(*args, **kwargs):
        result = original_choose(*args, **kwargs)
        if result[3]['no_jointly_feasible_candidate']:
            counters['joint_infeasibility_calls'] += 1
            result = extended_choose(*args, **kwargs)
            counters['extra_braking_selections'] += int(result[0][1] < -1)
        return result
    projection.choose_command = original_choose if arm == 'original' else emergency_choose
    env = make_env('ma', 'public_route_choice_fast_residual', guard_traffic=True)
    records, actions = [], []
    started = time.perf_counter()
    try:
        env.unwrapped._fixed_scenario = scenario
        obs, _ = env.reset(seed=2026)
        world = env.unwrapped
        assert world.speed_action.d_speed == 20/3
        assert world.d_heading == 45 and world.action_frequency == 10
        rng_before = repr(world._np_random.bit_generator.state)
        while env.agents:
            ids = list(env.agents)
            command = predict_actions(model, np.stack([obs[a] for a in ids]), batched=False)
            now = {a: float(world.metrics[a]['flight_time']) for a in ids}
            obs, _, terminated, truncated, infos = env.step(dict(zip(ids, command)))
            for agent in ids:
                diagnostic = world.last_projection[agent]
                actions.append(dict(time=now[agent], agent=agent, **diagnostic))
                if terminated[agent] or truncated[agent]:
                    records.append(dict(episode=0, agent=agent, **infos[agent]))
        assert repr(world._np_random.bit_generator.state) == rng_before
        result = dict(summary=summarize(records, 1, 10), records=records,
            wall_seconds=time.perf_counter()-started, counters=counters,
            traffic_statistics=dict(world.traffic_projection_statistics),
            emergency_actions=[a for a in actions if a['commanded_speed_action'] < -1])
        if arm == 'original':
            actual = {r['agent']:r for r in records}
            differences = {k:max(abs(expected[a][k]-actual[a][k]) for a in expected) for k in METRICS}
            assert not any(differences.values()), differences
            result['all_nine_reference_metrics_exact'] = True
            result['maximum_absolute_metric_differences'] = differences
        else:
            assert all(-3 <= a['commanded_speed_action'] <= 1 for a in actions)
            assert all(abs(a['commanded_heading_turn_deg']) <= 45 for a in actions)
        report['arms'][arm] = result
    finally:
        env.close()
        projection.choose_command = original_choose
    (OUTPUT / 'targeted-results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(arm=arm, summary=result['summary'], counters=counters)), flush=True)
