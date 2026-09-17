from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from atc_rl.policy import AircraftPolicy
from atc_rl.initialization import neutral_action_mean
from atc_rl.exploration import configuration, ppo_options, log_std_initialization, require_matching, validate_model


class ConstantObservation(Env):
    observation_space = spaces.Dict({"actor": spaces.Box(-1, 1, (3,), dtype=np.float32),
                                    "critic": spaces.Box(-1, 1, (8,), dtype=np.float32)})
    action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)

    def __init__(self):
        self.actions = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return {"actor": np.full(3, .2, np.float32), "critic": np.full(8, .4, np.float32)}, {}

    def step(self, action):
        self.actions.append(np.array(action).copy())
        return self.reset()[0], float(1 - np.square(action).sum()), False, False, {}


def make_model(centralized=False, frequency=2):
    config = {"exploration": "gsde", "sde_weight_std": .05, "sde_sample_freq": frequency,
              "initial_action_std": None}
    model = PPO(AircraftPolicy, ConstantObservation(), seed=24, n_steps=4, batch_size=4,
                n_epochs=1, device="cpu", **ppo_options(config),
                policy_kwargs={"centralized": centralized, "actor_width": 8, "critic_width": 8,
                               "log_std_init": log_std_initialization(config)})
    neutral_action_mean(model)
    return model, config


def test_legacy_configuration_remains_gaussian():
    assert configuration({}) == {"exploration": "gaussian", "sde_weight_std": None, "sde_sample_freq": None}
    assert ppo_options({}) == {"use_sde": False, "sde_sample_freq": -1}
    assert log_std_initialization({"initial_action_std": .05}) == np.log(.05)
    require_matching({}, {"exploration": "gaussian"})


@pytest.mark.parametrize("config", [
    {"exploration": "unknown"},
    {"sde_weight_std": .05},
    {"sde_sample_freq": 12},
    {"exploration": "gsde"},
    {"exploration": "gsde", "sde_weight_std": 0, "sde_sample_freq": 12},
    {"exploration": "gsde", "sde_weight_std": float("nan"), "sde_sample_freq": 12},
    {"exploration": "gsde", "sde_weight_std": .05, "sde_sample_freq": 0},
    {"exploration": "gsde", "sde_weight_std": .05, "sde_sample_freq": True},
])
def test_invalid_or_ambiguous_exploration_settings_are_rejected(config):
    with pytest.raises(ValueError):
        configuration(config)


@pytest.mark.parametrize("centralized", [False, True])
def test_gsde_mean_variance_and_sample_ignore_joint_context_and_reload(tmp_path, centralized):
    torch.set_num_threads(1)
    model, config = make_model(centralized)
    validate_model(model, config)
    own = torch.full((10, 3), .2)
    observation = {"actor": own, "critic": torch.zeros((10, 8), requires_grad=True)}
    model.policy.reset_noise(10)
    distribution = model.policy.get_distribution(observation)
    first = distribution.get_actions().detach().clone()
    std = distribution.distribution.stddev.detach().clone()
    mean = distribution.distribution.mean.detach().clone()
    assert torch.count_nonzero(mean) == 0
    changed_context = {"actor": own, "critic": torch.ones((10, 8))}
    other = model.policy.get_distribution(changed_context)
    assert torch.equal(first, other.get_actions())
    assert torch.equal(std, other.distribution.stddev)
    assert not torch.equal(first[0], first[1])
    assert not torch.allclose(std, torch.full_like(std, .05)), "Weight std is not the marginal action std"
    model.policy.reset_noise(10)
    assert not torch.equal(first, model.policy.get_distribution(observation).get_actions())
    path = tmp_path / "policy.zip"
    model.save(path)
    restored = PPO.load(path, device=model.device)
    validate_model(restored, config)
    a = model.policy.get_distribution(observation).distribution
    b = restored.policy.get_distribution(observation).distribution
    assert torch.equal(a.mean, b.mean) and torch.equal(a.stddev, b.stddev)
    with pytest.raises(ValueError, match="distribution"):
        validate_model(restored, {})
    with pytest.raises(ValueError, match="frequency"):
        validate_model(restored, {**config, "sde_sample_freq": 12})
    model.env.close()


def test_gsde_rollout_resamples_on_schedule_and_optimizer_updates_actor():
    model, config = make_model(frequency=2)
    before = model.policy.action_net.weight.detach().clone()
    model.learn(4)
    actions = model.env.envs[0].unwrapped.actions
    assert len(actions) == 4
    np.testing.assert_array_equal(actions[0], actions[1])
    np.testing.assert_array_equal(actions[2], actions[3])
    assert not np.array_equal(actions[1], actions[2])
    assert not torch.equal(before, model.policy.action_net.weight)
    assert torch.isfinite(model.policy.log_std).all()
    model.env.close()


def test_exploration_difference_cannot_be_mislabelled_as_critic_difference():
    a = {"exploration": "gsde", "sde_weight_std": .05, "sde_sample_freq": 12}
    for b in ({}, {**a, "sde_sample_freq": 1}, {**a, "sde_weight_std": .2}):
        with pytest.raises(ValueError, match="Exploration"):
            require_matching(a, b)


def test_explicit_legacy_options_preserve_initial_policy_tensors():
    def make(extra):
        return PPO(AircraftPolicy, ConstantObservation(), seed=27, n_steps=4, batch_size=4, n_epochs=1,
                   device="cpu", policy_kwargs={"actor_width": 8, "critic_width": 8, "log_std_init": np.log(.05)}, **extra)
    old, new = make({}), make(ppo_options({}))
    assert all(torch.equal(value, new.policy.state_dict()[key]) for key, value in old.policy.state_dict().items())
    old.env.close()
    new.env.close()
