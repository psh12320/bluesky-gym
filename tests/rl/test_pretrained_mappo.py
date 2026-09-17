import copy
import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from atc_rl.actor_reference import is_actor_parameter, match_initial_actor
from atc_rl.checkpoint_identity import policy_fingerprint
from atc_rl.demonstrations import sha256
from atc_rl.imitation import ObservationOnlyEnvironment, actor_parameters, fit_epoch
from atc_rl.maneuvers import ManeuverMapping, initialize_route_choice
from atc_rl.policy import AircraftPolicy
from atc_rl.pretrained import initialize_from_pretrained, audit_pretraining


def supervised_parent(directory, categorical):
    torch.set_num_threads(1)
    mapping = ManeuverMapping(3, 0, 1) if categorical else None
    environment = ObservationOnlyEnvironment(np.array([-1., -1., 0.]), np.ones(3), mapping)
    model = PPO(AircraftPolicy, environment, seed=93, device='cpu', n_steps=2, batch_size=2,
                policy_kwargs={'actor_width': 8, 'critic_width': 8, 'log_std_init': math.log(.05)})
    if categorical:
        initialize_route_choice(model)
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    observations = torch.tensor(np.tile(np.array([1., 0., .5], np.float32), (16, 1)))
    targets = torch.tensor(np.tile([19, 0], (16, 1))) if categorical else torch.full((16, 2), .4)
    optimizer = torch.optim.Adam(actor_parameters(model.policy), lr=.03)
    for _ in range(10):
        fit_epoch(model.policy, observations, targets, torch.ones(16), optimizer, np.arange(16), 16, mapping)
    assert any(not torch.equal(v, before[k]) for k, v in model.policy.state_dict().items() if is_actor_parameter(k))
    assert all(torch.equal(v, before[k]) for k, v in model.policy.state_dict().items() if not is_actor_parameter(k))
    directory.mkdir()
    checkpoint = directory / 'model.zip'
    model.save(checkpoint)
    config = {'algorithm': 'behavior_cloning', 'training_method': 'behavior_cloning_only',
              'reinforcement_learning_performed': False, 'epochs': 10,
              'guidance': True, 'filter': False, 'static_filter': True, 'conflict_features': True,
              'mask_conflict_features': False, 'traffic_position_scale': 1., 'action_reference': 'direct',
              'exploration': 'categorical' if categorical else 'gaussian',
              'initial_action_std': None if categorical else .05, 'actor_widths': [8, 8], 'critic_widths': [8, 8]}
    if mapping:
        config['maneuver_mapping'] = mapping.specification()
    summary = {'status': 'complete', 'critic_and_noise_unchanged': True, 'serialization_actions_identical': True,
               'reinforcement_learning_updates': 0, 'model_sha256': sha256(checkpoint), 'epochs': 10,
               'supervised_optimizer_steps': 10, 'supervised_examples_seen': 160, 'demonstration_transitions': 16}
    record = {'file': checkpoint.name, 'sha256': sha256(checkpoint), 'policy_fingerprint': policy_fingerprint(checkpoint),
              'live_transitions': 0, 'counted_transitions': 0, 'optimizer_steps': 0, 'supervised_epochs': 10,
              **{key: summary[key] for key in ('supervised_optimizer_steps', 'supervised_examples_seen', 'demonstration_transitions')}}
    for name, value in [('config.json', config), ('training_summary.json', summary), ('checkpoints.json', [record]),
                        ('provenance.json', {'synthetic_test_fixture': True})]:
        (directory / name).write_text(json.dumps(value), encoding='utf-8')
    return model, config, checkpoint


def target_model(parent, centralized):
    environment = ObservationOnlyEnvironment(parent.observation_space['actor'].low,
                                            parent.observation_space['actor'].high,
                                            ManeuverMapping(3, 0, 1) if hasattr(parent.action_space, 'nvec') else None)
    return PPO(AircraftPolicy, environment, seed=95, device='cpu', n_steps=2, batch_size=2,
               policy_kwargs={'centralized': centralized, 'actor_width': 8, 'critic_width': 8,
                              'log_std_init': math.log(.05)})


@pytest.mark.parametrize('categorical', [False, True])
@pytest.mark.parametrize('centralized', [False, True])
def test_supervised_transfer_keeps_actor_outputs_and_records_true_learning_history(tmp_path, categorical, centralized):
    parent, config, checkpoint = supervised_parent(tmp_path / 'parent', categorical)
    target = target_model(parent, centralized)
    before = {name: value.clone() for name, value in target.policy.state_dict().items()}
    config = {**config, 'algorithm': 'mappo' if centralized else 'ppo', 'neutral_action_mean': False,
              'pretrained_model': str(checkpoint)}
    directory = tmp_path / 'target'; directory.mkdir()
    torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    lineage = initialize_from_pretrained(target, checkpoint, config, directory)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert python_rng == random.getstate()
    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
    assert numpy_rng[0] == np.random.get_state()[0] and numpy_rng[2:] == np.random.get_state()[2:]
    assert not target.policy.optimizer.state and target.num_timesteps == target._n_updates == 0
    parent_state = parent.policy.state_dict()
    for name, value in target.policy.state_dict().items():
        expected = before[name] if centralized and not is_actor_parameter(name) else parent_state[name]
        assert torch.equal(value, expected), name
    inputs = {'actor': np.tile(np.array([1., 0., .5], np.float32), (5, 1)),
              'critic': np.ones((5, *target.observation_space['critic'].shape), np.float32)}
    np.testing.assert_array_equal(target.predict(inputs, deterministic=True)[0], parent.predict(inputs, deterministic=True)[0])
    inputs['critic'] *= 17
    np.testing.assert_array_equal(target.predict(inputs, deterministic=True)[0], parent.predict(inputs, deterministic=True)[0])
    context = torch.ones((1, *target.observation_space['critic'].shape), requires_grad=True)
    value = target.policy.predict_values({'actor': torch.ones((1, 3)), 'critic': context})
    gradient = torch.autograd.grad(value.sum(), context, allow_unused=True)[0]
    assert bool(gradient is not None and torch.count_nonzero(gradient)) == centralized
    target.save(directory / 'initial-model.zip')
    config['pretraining'] = lineage
    assert audit_pretraining(directory, config) == lineage
    assert lineage['supervised_optimizer_steps'] == 10 and not lineage['initial_policy_is_untrained']
    if centralized:
        assert lineage['transfer_scope'] == 'actor_only'
        assert policy_fingerprint(directory / 'initial-model.zip') != policy_fingerprint(checkpoint)
    else:
        assert policy_fingerprint(directory / 'initial-model.zip') == policy_fingerprint(checkpoint)
    parent.env.close(); target.env.close()


@pytest.mark.parametrize('tamper', ['actor', 'critic', 'scope', 'algorithm'])
def test_mappo_lineage_audit_rejects_changed_initialization(tmp_path, tamper):
    parent, config, checkpoint = supervised_parent(tmp_path / 'parent', True)
    target = target_model(parent, True)
    config = {**config, 'algorithm': 'mappo', 'neutral_action_mean': False, 'pretrained_model': str(checkpoint)}
    directory = tmp_path / 'target'; directory.mkdir()
    config['pretraining'] = initialize_from_pretrained(target, checkpoint, config, directory)
    if tamper in ('actor', 'critic'):
        parameter = next(p for name, p in target.policy.named_parameters()
                         if is_actor_parameter(name) == (tamper == 'actor'))
        with torch.no_grad(): parameter.add_(.1)
    elif tamper == 'scope':
        config['pretraining']['transfer_scope'] = 'full_policy'
    else:
        config['algorithm'] = 'ppo'
    target.save(directory / 'initial-model.zip')
    with pytest.raises(ValueError, match='starting actor|starting critic|transfer scope'):
        audit_pretraining(directory, config)
    parent.env.close(); target.env.close()


@pytest.mark.parametrize('marker', ['pretraining', 'pretrained_model', 'checkpoint_pretraining', 'supervised_optimizer_steps'])
def test_untrained_actor_reference_rejects_supervised_history(tmp_path, marker):
    helper_path = Path(__file__).with_name('test_actor_reference.py')
    spec = importlib.util.spec_from_file_location('actor_reference_test_helpers', helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    reference = helper.make_model(False)
    checkpoint = helper.save_reference(reference, tmp_path / 'reference')
    if marker in ('pretraining', 'pretrained_model'):
        path = checkpoint.parent / 'config.json'; data = json.loads(path.read_text())
        data[marker] = {'method': 'behavior_cloning'} if marker == 'pretraining' else 'supervised/model.zip'
    else:
        path = checkpoint.parent / 'checkpoints.json'; data = json.loads(path.read_text())
        data[0]['pretraining' if marker == 'checkpoint_pretraining' else marker] = {} if marker == 'checkpoint_pretraining' else 10
    path.write_text(json.dumps(data))
    target = helper.make_model(True)
    before = {name: value.clone() for name, value in target.policy.state_dict().items()}
    with pytest.raises(ValueError, match='explicit pretraining lineage'):
        match_initial_actor(target, checkpoint, tmp_path / 'copied')
    assert not (tmp_path / 'copied').exists()
    assert all(torch.equal(value, before[name]) for name, value in target.policy.state_dict().items())
    reference.env.close(); target.env.close()
