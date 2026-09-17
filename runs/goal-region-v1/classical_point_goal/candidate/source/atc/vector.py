import numpy as np
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
from pettingzoo.utils import BaseParallelWrapper


class SeededVecEnv(SB3VecEnvWrapper):
    """Apply SB3's pending seed to the first simulator reset exactly once."""

    def __init__(self, env):
        super().__init__(env)
        self._pending_seed = None

    def seed(self, seed=None):
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        self._pending_seed = int(seed)
        return [int(seed) + i for i in range(self.num_envs)]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        observations, self.reset_infos = self.venv.reset(seed=self._pending_seed, options=options)
        self._pending_seed = None
        return observations


class FixedPopulation(BaseParallelWrapper):
    """Keep completed aircraft as absorbing zero-reward slots until world reset."""

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        self._finished = {}
        self._zeros = {
            a: np.zeros(self.observation_space(a).shape, dtype=self.observation_space(a).dtype)
            for a in self.possible_agents
        }
        return observations, infos

    def step(self, actions):
        active = list(self.env.agents)
        observations, rewards, terms, truncs, infos = self.env.step({a: actions[a] for a in active})
        for a in active:
            if terms[a] or truncs[a]:
                self._finished[a] = (terms[a], truncs[a])
        world_done = not self.env.agents
        padded_obs, padded_rewards, padded_infos, out_terms, out_truncs = {}, {}, {}, {}, {}
        for a in self.possible_agents:
            padded_obs[a] = observations.get(a, self._zeros[a]).copy()
            padded_rewards[a] = rewards.get(a, 0.0)
            padded_infos[a] = dict(infos.get(a, {}))
            padded_infos[a]["inactive"] = a not in active
            padded_infos[a]["aircraft_terminated"] = bool(terms.get(a, False))
            padded_infos[a]["aircraft_truncated"] = bool(truncs.get(a, False))
            terminal, timeout = self._finished.get(a, (False, False))
            out_terms[a] = bool(world_done and terminal)
            out_truncs[a] = bool(world_done and timeout and not terminal)
            padded_infos[a]["TimeLimit.truncated"] = out_truncs[a]
        if world_done:
            self.agents = []
        return padded_obs, padded_rewards, out_terms, out_truncs, padded_infos
