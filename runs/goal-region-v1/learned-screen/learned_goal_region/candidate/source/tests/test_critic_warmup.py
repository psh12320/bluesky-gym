import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.logger import configure

from atc.algorithms import NavigationSAC, initialize_navigation_actor


class ReplayEnv(gym.Env):
    observation_space = spaces.Box(-1., 1., (4,), dtype=np.float32)
    action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)


def make_model(tmp_path, held):
    torch.set_num_threads(1)
    model = NavigationSAC('MlpPolicy', ReplayEnv(), buffer_size=32,
                          policy_kwargs={'net_arch': [16, 16]}, seed=2800,
                          critic_warmup_updates=held)
    initialize_navigation_actor(model)
    model.set_logger(configure(str(tmp_path), []))
    rng = np.random.default_rng(2800)
    for _ in range(32):
        model.replay_buffer.add(rng.uniform(-1, 1, (1, 4)).astype(np.float32),
            rng.uniform(-1, 1, (1, 4)).astype(np.float32),
            rng.uniform(-1, 1, (1, 2)).astype(np.float32),
            rng.normal(size=1).astype(np.float32), np.zeros(1), [{}])
    return model


def snapshot(module):
    return {key: value.clone() for key, value in module.state_dict().items()}


def unchanged(before, module):
    return all(torch.equal(value, module.state_dict()[key]) for key, value in before.items())


def test_actor_held_while_critics_and_temperature_learn_then_released(tmp_path):
    model = make_model(tmp_path, 2)
    actor, critic, target = snapshot(model.actor), snapshot(model.critic), snapshot(model.critic_target)
    temperature = model.log_ent_coef.clone()
    model.train(2, 8)
    assert model._n_updates == 2
    assert unchanged(actor, model.actor)
    assert not unchanged(critic, model.critic)
    assert not unchanged(target, model.critic_target)
    assert not torch.equal(temperature, model.log_ent_coef)
    assert all(parameter.requires_grad for parameter in model.actor.parameters())
    model.train(1, 8)
    assert model._n_updates == 3
    assert not unchanged(actor, model.actor)
    assert model.logger.name_to_value['train/actor_updates'] == 1


def test_warmup_boundary_and_resume_keep_remaining_update_count(tmp_path):
    model = make_model(tmp_path, 2)
    actor = snapshot(model.actor)
    model.train(1, 8)
    model.save(tmp_path / 'model')
    model.save_replay_buffer(tmp_path / 'replay.pkl')
    loaded = NavigationSAC.load(tmp_path / 'model.zip', env=ReplayEnv())
    loaded.load_replay_buffer(tmp_path / 'replay.pkl')
    loaded.set_logger(configure(str(tmp_path), []))
    assert loaded.critic_warmup_updates == 2
    assert loaded._n_updates == 1
    loaded.train(1, 8)
    assert unchanged(actor, loaded.actor)
    loaded.train(1, 8)
    assert not unchanged(actor, loaded.actor)
    boundary = make_model(tmp_path, 2)
    initial = snapshot(boundary.actor)
    boundary.train(3, 8)
    assert boundary._n_updates == 3
    assert not unchanged(initial, boundary.actor)
    assert boundary.logger.name_to_value['train/held_actor_updates'] == 2
    assert boundary.logger.name_to_value['train/actor_updates'] == 1


def test_legacy_checkpoint_defaults_to_no_actor_hold(tmp_path):
    model = make_model(tmp_path, 0)
    del model.critic_warmup_updates
    model.save(tmp_path / 'legacy')
    loaded = NavigationSAC.load(tmp_path / 'legacy.zip', env=ReplayEnv())
    loaded.set_logger(configure(str(tmp_path), []))
    assert loaded.critic_warmup_updates == 0
    loaded.replay_buffer = model.replay_buffer
    actor = snapshot(loaded.actor)
    loaded.train(1, 8)
    assert not unchanged(actor, loaded.actor)


@pytest.mark.parametrize('value', [-1, 1.5])
def test_invalid_warmup_rejected(tmp_path, value):
    with pytest.raises(ValueError, match='nonnegative integer'):
        make_model(tmp_path, value)
