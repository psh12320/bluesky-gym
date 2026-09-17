"""Potential-based navigation feedback with the original scoring records intact.

F(s,t,s',t') = gamma * Phi(s',t') - Phi(s,t). Phi is zero at every
actual task terminal, including the competition deadline. References:
Ng, Harada & Russell (ICML 1999); Grzes (AAMAS 2017, pp. 565-573).
This is a training aid, not a guarantee about approximate PPO performance.
"""
import math
from pettingzoo.utils import BaseParallelWrapper

GAMMA = 0.996508469331006
DISTANCE_NORM_KM = 150.0
DISTANCE_CAP_KM = 1500.0
GOAL_RADIUS_KM = 5.0


def potential(distance_km, remaining_fraction, scale):
    if not all(math.isfinite(v) for v in (distance_km, remaining_fraction, scale)):
        raise ValueError('Potential inputs must be finite')
    if distance_km < 0 or scale < 0 or not 0 <= remaining_fraction <= 1:
        raise ValueError('Invalid distance, time fraction or shaping scale')
    remaining_distance = min(max(distance_km-GOAL_RADIUS_KM, 0.0), DISTANCE_CAP_KM)
    # Time is already part of the actor/critic state. This taper avoids a large
    # one-step bonus at the actual timeout while keeping the telescoping identity.
    return -scale * remaining_distance / DISTANCE_NORM_KM * remaining_fraction


def shaping_reward(previous_potential, next_potential, terminal, gamma=GAMMA):
    if not 0 < gamma <= 1 or not all(math.isfinite(v) for v in (previous_potential,next_potential)):
        raise ValueError('Invalid discount or potential')
    return gamma * (0.0 if terminal else next_potential) - previous_potential


class PotentialProgress(BaseParallelWrapper):
    """Shape returned learning rewards; keep actions, observations and info intact."""
    def __init__(self, env, scale, gamma=GAMMA):
        super().__init__(env)
        potential(0.0,1.0,scale)
        if not 0 < gamma <= 1:raise ValueError('Invalid discount')
        self.scale=float(scale)
        self.gamma=float(gamma)
        self.potentials={}
        self.last_native_rewards={}
        self.last_shaping={}

    def _potential(self, agent):
        import bluesky as bs
        from core.tools import kwikqdrdist
        world=self.unwrapped
        index=bs.traf.id2idx(agent)
        if index < 0:raise RuntimeError('Live aircraft is absent from the simulator')
        goal_lat,goal_lon=world._goal[agent]
        _,distance_nm=kwikqdrdist(bs.traf.lat[index],bs.traf.lon[index],goal_lat,goal_lon)
        remaining=max(0.0,1.0-world.sim_time/world.episode_time_limit)
        return potential(float(distance_nm)*1.852,remaining,self.scale)

    def reset(self,seed=None,options=None):
        observation,info=self.env.reset(seed=seed,options=options)
        self.potentials={agent:self._potential(agent) for agent in self.env.agents}
        self.last_native_rewards={}
        self.last_shaping={}
        return observation,info

    def step(self,actions):
        active=list(self.env.agents)
        observation,native,terminated,truncated,info=self.env.step(actions)
        rewards={}
        next_potentials={}
        self.last_native_rewards={agent:float(native[agent]) for agent in active}
        self.last_shaping={}
        for agent in active:
            done=bool(terminated[agent] or truncated[agent])
            following=0.0 if done else self._potential(agent)
            bonus=shaping_reward(self.potentials[agent],following,done,self.gamma)
            self.last_shaping[agent]=bonus
            rewards[agent]=native[agent]+bonus
            if not done:next_potentials[agent]=following
        self.potentials=next_potentials
        return observation,rewards,terminated,truncated,info
