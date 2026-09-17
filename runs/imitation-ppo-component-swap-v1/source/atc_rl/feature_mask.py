"""A same-size zero-input control for the conflict-observation experiment."""
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper
from atc.conflicts import FEATURE_NAMES

class MaskConflictFeatures(BaseParallelWrapper):
    """Zero only added conflict features, retaining spaces, actions and scoring."""
    def __init__(self, environment):
        super().__init__(environment)
        self.indices = {}
        for agent in environment.possible_agents:
            dictionary = environment.unwrapped.observation_space(agent)
            if not isinstance(dictionary, spaces.Dict) or not set(FEATURE_NAMES).issubset(dictionary.spaces):
                raise ValueError("Conflict-feature masking requires the augmented observation")
            offsets = []
            cursor = 0
            for name, space in dictionary.spaces.items():
                size = spaces.flatdim(space)
                if name in FEATURE_NAMES:
                    offsets.extend(range(cursor, cursor + size))
                cursor += size
            if environment.observation_space(agent).shape != (cursor,):
                raise ValueError("Apply conflict-feature masking after observation flattening")
            self.indices[agent] = np.asarray(offsets, dtype=np.int64)

    def _mask(self, observations):
        masked = {}
        for agent, value in observations.items():
            result = np.asarray(value).copy()
            result[self.indices[agent]] = 0
            masked[agent] = result
        return masked

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._mask(observations), infos

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        return self._mask(observations), rewards, terminated, truncated, infos
