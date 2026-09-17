import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.buffers import ReplayBuffer


class AircraftReplayBuffer(ReplayBuffer):
    """Store live aircraft transitions, ending returns at each aircraft's arrival.

    Worlds reset together, but aircraft finish at different times. Absorbing
    padding is useful for vectorization; it must not become SAC experience.
    """

    def __init__(self, buffer_size, observation_space, action_space, device="auto",
                 n_envs=1, optimize_memory_usage=False, **kwargs):
        if optimize_memory_usage:
            raise ValueError("Aircraft replay requires separate next observations")
        self.input_envs = n_envs
        self.live_transitions = 0
        self.skipped_transitions = 0
        # SB3 policies consume float32 observations. Store that precision directly.
        storage_space = Box(observation_space.low.astype(np.float32),
                            observation_space.high.astype(np.float32), dtype=np.float32)
        super().__init__(buffer_size, storage_space, action_space, device=device,
                         n_envs=1, optimize_memory_usage=False, **kwargs)

    def add(self, obs, next_obs, action, reward, done, infos):
        for index, info in enumerate(infos):
            if info.get("inactive", False):
                self.skipped_transitions += 1
                continue
            terminal = bool(info.get("aircraft_terminated", done[index]))
            timeout = bool(info.get("aircraft_truncated", info.get("TimeLimit.truncated", False)))
            row_info = dict(info, **{"TimeLimit.truncated": timeout and not terminal})
            super().add(
                obs[index:index + 1], next_obs[index:index + 1],
                action[index:index + 1], reward[index:index + 1],
                np.array([terminal or timeout]), [row_info],
            )
            self.live_transitions += 1
