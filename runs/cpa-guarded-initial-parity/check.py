import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atc.compare import load_evaluation
from atc.envs import make_env
from atc.inference import predict_actions
from atc.metrics import METRICS
from atc.submission import _validate_configuration

torch.set_num_threads(1)
directories = [ROOT / p for p in (
    'runs/public-projected-v4-391k',
    'runs/public-cpa-guarded-initial-predictions',
    'runs/public-cpa-guarded-initial-control',
)]
models = []
for directory in directories:
    configuration = json.loads((directory/'config.json').read_text())
    _validate_configuration('ma', configuration)
    models.append(SAC.load(directory/'model.zip', device='cpu', buffer_size=1))
base, treatment, control = models
indices = treatment.actor.features_extractor.original_indices.cpu().numpy()
a, b = treatment.policy.state_dict(), control.policy.state_dict()
assert a.keys() == b.keys()
assert all(torch.equal(a[k], b[k]) for k in a)
assert treatment.actor.features_extractor.projection.weight.requires_grad
assert not control.actor.features_extractor.projection.weight.requires_grad
assert all(len(m.actor.optimizer.state)==0 and len(m.critic.optimizer.state)==0 for m in (treatment,control))
assert all(m.num_timesteps == 391520 for m in models)
_, old_rows = load_evaluation(ROOT/'runs/sac-ma-public-600k-seed1400/validation-projected-v4-individual-20')
expected = {(r['episode'],r['agent']):r for r in old_rows if r['episode'] < 2}
observed, checked_actions = {}, 0
env = make_env('ma','public_cpa',True)
try:
    for episode in range(2):
        observation, _ = env.reset(seed=2026 if episode == 0 else None)
        while env.agents:
            agents = list(env.agents)
            x = np.stack([observation[a] for a in agents])
            action = predict_actions(base, x[:,indices])
            for other in (treatment,control):
                assert np.array_equal(action,predict_actions(other,x))
            checked_actions += len(agents)
            observation, _, term, trunc, info = env.step(dict(zip(agents,action)))
            for agent in info:
                if term[agent] or trunc[agent]:
                    observed[(episode,agent)] = info[agent]
        print(f'Checked initial policy parity on scenario {episode+1}/2',flush=True)
finally:
    env.close()
assert observed.keys() == expected.keys()
assert all(observed[k][m] == expected[k][m] for k in observed for m in METRICS)
result = dict(episodes=2,seed=2026,checked_aircraft_decisions=checked_actions,
              identical_initial_actions=True,identical_policy_state_tensors=True,
              initial_optimizers_empty=True,exact_reference_metrics=list(METRICS),
              prediction_trainability=[True,False],
              model_sha256={str(p.relative_to(ROOT)):hashlib.sha256((p/'model.zip').read_bytes()).hexdigest() for p in directories})
(Path(__file__).parent/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
