"""Invertible rescaling of existing relative-position inputs; scoring stays native."""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper


def position_scale(config):
    value = config.get("traffic_position_scale", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Traffic position scale must be finite and positive")
    return float(value)


def require_matching(left, right):
    if position_scale(left) != position_scale(right):
        raise ValueError("Traffic position scaling differs")


class PositionTransform:
    def __init__(self, dictionary, flattened, scale):
        self.scale = position_scale({"traffic_position_scale": scale})
        if not isinstance(dictionary, spaces.Dict) or not {"x_r", "y_r"}.issubset(dictionary.spaces):
            raise ValueError("Expected relative x_r/y_r traffic positions")
        indices = []
        cursor = 0
        for name, space in dictionary.spaces.items():
            size = spaces.flatdim(space)
            if name in ("x_r", "y_r"):
                indices.extend(range(cursor, cursor + size))
            cursor += size
        if not isinstance(flattened, spaces.Box) or flattened.shape != (cursor,):
            raise ValueError("Scale positions after flattening and before adding the clock")
        self.indices = np.asarray(indices, dtype=np.int64)
        lower, upper = flattened.low.copy(), flattened.high.copy()
        lower[self.indices] *= self.scale
        upper[self.indices] *= self.scale
        self.space = spaces.Box(lower, upper, dtype=flattened.dtype)

    def __call__(self, observation):
        result = np.asarray(observation).copy()
        result[self.indices] *= self.scale
        if not np.isfinite(result[self.indices]).all():
            raise ValueError("Nonfinite scaled relative position")
        return result


class ScaleTrafficMA(BaseParallelWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.transforms = {a: PositionTransform(env.unwrapped.observation_space(a),
                            env.observation_space(a), scale) for a in env.possible_agents}

    def observation_space(self, agent):
        return self.transforms[agent].space

    def _observe(self, observations):
        return {a: self.transforms[a](value) for a, value in observations.items()}

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._observe(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._observe(observations), rewards, terminations, truncations, infos


class ScaleTrafficSA(gym.ObservationWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.transform = PositionTransform(env.unwrapped.observation_space, env.observation_space, scale)
        self.observation_space = self.transform.space

    def observation(self, observation):
        return self.transform(observation)
