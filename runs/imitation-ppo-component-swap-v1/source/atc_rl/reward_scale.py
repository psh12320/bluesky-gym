"""Scale learning rewards without changing simulator scoring or actions."""
import math
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class ScaledLearningRewards(VecEnvWrapper):
    """A fixed positive scale for value-target conditioning.

    Native metrics remain in simulator units. The episode return consumed by the
    training callback is scaled consistently with the rewards stored for PPO.
    This changes optimization, including its balance with entropy regularization;
    it is an experimental setting, not a promise of improved learning.
    """
    def __init__(self, environment, scale):
        scale=float(scale)
        if not math.isfinite(scale) or scale<=0:
            raise ValueError('Learning reward scale must be finite and positive')
        super().__init__(environment)
        self.scale=scale

    def reset(self):
        observations=self.venv.reset()
        self.reset_infos=self.venv.reset_infos
        return observations

    def step_wait(self):
        observations,rewards,dones,original_infos=self.venv.step_wait()
        scaled=np.asarray(rewards,dtype=np.float32)*self.scale
        if not np.isfinite(scaled).all():
            raise ValueError('Scaled learning rewards are non-finite')
        infos=[]
        for original in original_infos:
            info=dict(original)
            if 'learning_episode_return' in info:
                info['unscaled_learning_episode_return']=info['learning_episode_return']
                info['learning_episode_return']*=self.scale
            if 'episode' in info:
                info['episode']={**info['episode'],'r':info['episode']['r']*self.scale}
            infos.append(info)
        self.reset_infos=self.venv.reset_infos
        return observations,scaled,dones,infos
