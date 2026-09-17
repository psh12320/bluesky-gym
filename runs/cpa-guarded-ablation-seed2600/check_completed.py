"""Verify completed budgets and prediction-branch training for the paired pilot."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from stable_baselines3 import SAC
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
keys = ('env', 'algorithm', 'recipe', 'guard_static', 'workers', 'seed', 'world_seeds',
        'gamma', 'reward_kwargs', 'observation_shape', 'actual_net_arch',
        'actual_learning_rate', 'actual_tau', 'actual_gradient_steps',
        'actual_buffer_size', 'actual_learning_starts', 'actual_batch_size',
        'initial_timesteps', 'static_projection_revision',
        'static_projection_horizon_seconds', 'static_projection_clearance_km',
        'conflict_prediction_horizon_seconds')
configs, records = [], []
for variant in ('control', 'predictions'):
    path = ROOT / f'runs/sac-ma-cpa-guarded-{variant}-seed2600'
    config = json.loads((path / 'config.json').read_text())
    training = json.loads((path / 'training_summary.json').read_text())
    model = SAC.load(path / 'model.zip', device='cpu', buffer_size=1)
    projections = {key: value for key, value in model.policy.state_dict().items() if key.endswith('projection.weight')}
    assert len(projections) == 3
    enabled = variant == 'predictions'
    assert model.actor.features_extractor.projection.weight.requires_grad == enabled
    assert model.critic.features_extractor.projection.weight.requires_grad == enabled
    assert all(bool(torch.count_nonzero(value)) == enabled for value in projections.values())
    assert model.num_timesteps == training['timesteps'] == 491520
    assert training['new_timesteps'] == 100000
    assert training['new_live_transitions'] + training['new_skipped_transitions'] == 100000
    configs.append(config)
    records.append(dict(variant=variant, training=training, num_updates=model._n_updates,
        model_sha256=hashlib.sha256((path / 'model.zip').read_bytes()).hexdigest(),
        prediction_projection_trainable=enabled,
        projection_norms={k: float(torch.linalg.vector_norm(v)) for k, v in projections.items()}))
    del model
assert all(configs[0][key] == configs[1][key] for key in keys)
assert records[0]['num_updates'] == records[1]['num_updates']
result = dict(matched_configuration_fields=list(keys), matched_counted_budget=True,
              matched_optimizer_update_count=True, records=records,
              note='Live transitions differ because the evolving policies produce different trajectories.')
(Path(__file__).parent / 'completion_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
