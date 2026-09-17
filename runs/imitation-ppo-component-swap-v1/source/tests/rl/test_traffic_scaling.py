import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import numpy as np
import pytest
from atc.envs import FlattenObservations
from atc_rl.traffic_scaling import PositionTransform, ScaleTrafficMA, ScaleTrafficSA, position_scale, require_matching
from atc_rl.curves import validate_configuration
from atc_rl.world_worker import pack_observations


def dictionary():
    return spaces.Dict({"x_r": spaces.Box(-2., 2., (2,), dtype=np.float64),
                        "y_r": spaces.Box(-2., 2., (2,), dtype=np.float64),
                        "traffic_dcpa": spaces.Box(0., 4., (2,), dtype=np.float64),
                        "cos_drift": spaces.Box(-1., 1., (1,), dtype=np.float64)})


def values():
    return {"x_r": np.array([.03, 0.]), "y_r": np.array([-.01, 0.]),
            "traffic_dcpa": np.array([.2, 0.]), "cos_drift": np.array([1.])}


class Multi(ParallelEnv):
    possible_agents = ["a", "b"]
    def __init__(self):
        self.agents = self.possible_agents.copy()
        self.dictionary = dictionary()
        self.native = ({"a": 2., "b": -1.}, {"a": False, "b": True},
                       {"a": False, "b": False}, {"a": {"score": 12}, "b": {"score": 3}})
    def observation_space(self, agent): return self.dictionary
    def action_space(self, agent): return spaces.Box(-1., 1., (2,), dtype=np.float32)
    def reset(self, seed=None, options=None):
        self.reset_infos = {"a": {"marker": 3}, "b": {}}
        return {a: values() for a in self.agents}, self.reset_infos
    def step(self, actions):
        self.received = actions
        return {a: values() for a in self.agents}, *self.native


class Single(gym.Env):
    observation_space = dictionary()
    action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
    def reset(self, seed=None, options=None): return values(), {"marker": 3}
    def step(self, action):
        self.received = action
        return values(), 7., False, True, {"native_score": 5}


def test_transform_only_rescales_positions_and_retains_padding_and_bounds():
    space = dictionary()
    flat = spaces.flatten(space, values())
    before = flat.copy()
    transform = PositionTransform(space, spaces.flatten_space(space), 20)
    result = transform(flat)
    np.testing.assert_array_equal(flat, before)
    observed = spaces.unflatten(space, result)
    np.testing.assert_allclose(observed["x_r"], [.6, 0.])
    np.testing.assert_allclose(observed["y_r"], [-.2, 0.])
    np.testing.assert_array_equal(observed["traffic_dcpa"], values()["traffic_dcpa"])
    np.testing.assert_array_equal(observed["cos_drift"], values()["cos_drift"])
    np.testing.assert_array_equal(transform.space.high[transform.indices], np.full(4, 40.))
    assert transform.space.contains(result)
    np.testing.assert_allclose(result[transform.indices] / 20, flat[transform.indices])


def test_multi_wrapper_preserves_native_actions_rewards_terminals_and_info():
    world = Multi()
    env = ScaleTrafficMA(FlattenObservations(world), 20)
    obs, infos = env.reset()
    assert infos is world.reset_infos
    actions = {a: np.array([.2, -.1]) for a in world.agents}
    result = env.step(actions)
    assert world.received is actions
    assert all(result[i+1] is part for i, part in enumerate(world.native))
    assert all(env.observation_space(a).contains(value) for a, value in result[0].items())
    dim = len(obs["a"])
    packed = pack_observations(obs, world.agents, world.possible_agents, dim, 1.)
    expected = packed["actor"].reshape(-1)
    np.testing.assert_array_equal(packed["critic"][0, :len(expected)], expected)
    np.testing.assert_array_equal(packed["critic"][1, :len(expected)], expected)


def test_single_and_multi_wrappers_use_the_same_scaling_and_preserve_scoring():
    single = Single()
    env = ScaleTrafficSA(gym.wrappers.FlattenObservation(single), 20)
    obs, _ = env.reset()
    multi, _ = ScaleTrafficMA(FlattenObservations(Multi()), 20).reset()
    np.testing.assert_array_equal(obs, multi["a"])
    action = np.array([.1, .3])
    output = env.step(action)
    assert single.received is action
    assert output[1:] == (7., False, True, {"native_score": 5})


def test_factor_one_is_bitwise_identity():
    space = dictionary()
    raw = spaces.flatten(space, values())
    np.testing.assert_array_equal(PositionTransform(space, spaces.flatten_space(space), 1)(raw), raw)
    require_matching({}, {"traffic_position_scale": 1.0})


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "20"])
def test_invalid_scales_are_rejected(value):
    with pytest.raises(ValueError, match="positive"):
        position_scale({"traffic_position_scale": value})


def test_wrong_wrapper_order_or_missing_positions_is_rejected():
    space = dictionary()
    with pytest.raises(ValueError, match="flattening"):
        PositionTransform(space, space, 20)
    with pytest.raises(ValueError, match="x_r"):
        PositionTransform(spaces.Dict({}), spaces.Box(-1., 1., (1,)), 20)


def test_comparisons_cannot_silently_mix_coordinate_scales():
    with pytest.raises(ValueError, match="scaling"):
        require_matching({}, {"traffic_position_scale": 20})
    with pytest.raises(ValueError, match="scaling"):
        validate_configuration({"traffic_position_scale": 20}, {}, {}, 0)
