import numpy as np
from gymnasium.spaces import Box

from atc.vector import SeededVecEnv


class SeedProbe:
    num_envs = 2
    observation_space = Box(-1.0, 1.0, (1,))
    action_space = Box(-1.0, 1.0, (1,))
    render_mode = None

    def __init__(self):
        self.seeds = []

    def reset(self, seed=None, options=None):
        self.seeds.append(seed)
        return np.zeros((2, 1)), [{}, {}]


def test_model_seed_reaches_simulator_once_then_rng_stream_continues():
    probe = SeedProbe()
    env = SeededVecEnv(probe)
    env.seed(17)
    env.reset()
    env.reset()
    env.reset(seed=23)
    assert probe.seeds == [17, None, 23]


from pettingzoo import ParallelEnv
from atc.vector import FixedPopulation
from supersuit.vector import MarkovVectorEnv


class TwoAircraft(ParallelEnv):
    possible_agents = ["a", "b"]
    metadata = {"name": "two_aircraft", "render_modes": []}
    render_mode = None

    def __init__(self):
        self.resets = 0

    def observation_space(self, agent):
        return Box(-1.0, 1.0, (1,))

    def action_space(self, agent):
        return Box(-1.0, 1.0, (1,))

    def reset(self, seed=None, options=None):
        self.resets += 1
        self.agents = ["a", "b"]
        self.steps = 0
        return {a: np.zeros(1, dtype=np.float32) for a in self.agents}, {a: {} for a in self.agents}

    def step(self, actions):
        assert set(actions) == set(self.agents)
        self.steps += 1
        if self.steps == 1:
            self.agents = ["b"]
            return ({"a": np.ones(1, dtype=np.float32), "b": np.zeros(1, dtype=np.float32)},
                    {"a": 5.0, "b": 0.0}, {"a": True, "b": False},
                    {"a": False, "b": False}, {"a": {}, "b": {}})
        self.agents = []
        return ({"b": np.ones(1, dtype=np.float32)}, {"b": -1.0}, {"b": False},
                {"b": True}, {"b": {}})


def test_mixed_goal_and_timeout_reset_without_an_extra_empty_step():
    world = TwoAircraft()
    vec = MarkovVectorEnv(FixedPopulation(world))
    vec.reset(seed=17)
    _, reward, terms, truncs, _ = vec.step(np.zeros((2, 1)))
    assert reward.tolist() == [5.0, 0.0]
    assert not (terms | truncs).any()
    _, reward, terms, truncs, infos = vec.step(np.zeros((2, 1)))
    assert world.resets == 2
    assert terms.tolist() == [True, False]
    assert truncs.tolist() == [False, True]
    assert infos[1]["TimeLimit.truncated"]
    assert infos[1]["terminal_observation"].tolist() == [1.0]
    assert reward.tolist() == [0.0, -1.0]
