from numbers import Integral

import numpy as np
import torch
from stable_baselines3 import SAC


class NavigationSAC(SAC):
    """Explore near a zero-action navigation reference during replay warmup."""

    def __init__(self, *args, navigation_warmup_std=0.15, critic_warmup_updates=0, **kwargs):
        if not isinstance(critic_warmup_updates, Integral) or critic_warmup_updates < 0:
            raise ValueError("critic_warmup_updates must be a nonnegative integer")
        self.critic_warmup_updates = int(critic_warmup_updates)
        self.navigation_warmup_std = navigation_warmup_std
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps, batch_size=64):
        """Hold actor parameters initially; critics and entropy temperature still learn."""
        held = min(gradient_steps, max(0, self.critic_warmup_updates - self._n_updates))
        if held:
            parameters = list(self.actor.parameters())
            original_flags = [parameter.requires_grad for parameter in parameters]
            try:
                for parameter in parameters:
                    parameter.requires_grad_(False)
                super().train(held, batch_size)
            finally:
                for parameter, flag in zip(parameters, original_flags):
                    parameter.requires_grad_(flag)
        if gradient_steps > held:
            super().train(gradient_steps - held, batch_size)
        self.logger.record("train/actor_updates", max(0, self._n_updates - self.critic_warmup_updates))
        self.logger.record("train/held_actor_updates", min(self._n_updates, self.critic_warmup_updates))

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            normalized = np.clip(np.random.normal(
                0.0, self.navigation_warmup_std, (n_envs, *self.action_space.shape)), -1.0, 1.0).astype(np.float32)
            return self.policy.unscale_action(normalized), normalized
        return super()._sample_action(learning_starts, action_noise, n_envs)


def initialize_navigation_actor(model):
    """Start at the known route-following mean with modest exploration noise."""
    with torch.no_grad():
        model.actor.mu.weight.zero_()
        model.actor.mu.bias.zero_()
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(-2.0)
