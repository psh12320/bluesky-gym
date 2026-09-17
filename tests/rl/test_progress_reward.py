import numpy as np
import pytest
from pettingzoo import ParallelEnv
from atc_rl.progress_reward import GAMMA, PotentialProgress, potential, shaping_reward


@pytest.mark.parametrize('length',[1,19,64,600])
def test_discounted_shaping_changes_complete_return_only_by_initial_potential(length):
    distance=np.random.default_rng(31).uniform(5,900,length+1)
    phi=[potential(float(d),1-i/length,100) for i,d in enumerate(distance)]
    bonuses=[shaping_reward(phi[i],phi[i+1],i==length-1) for i in range(length)]
    assert sum(GAMMA**i*b for i,b in enumerate(bonuses))==pytest.approx(-phi[0],abs=1e-9)


def test_rollout_cut_keeps_the_nonterminal_potential():
    phi=[potential(180-i,1-i/600,100) for i in range(65)]
    bonuses=[shaping_reward(phi[i],phi[i+1],False) for i in range(64)]
    assert sum(GAMMA**i*b for i,b in enumerate(bonuses))==pytest.approx(GAMMA**64*phi[-1]-phi[0],abs=1e-9)


def test_task_terminal_overrides_nonzero_next_potential():
    assert shaping_reward(-7,-100,True)==7
    assert shaping_reward(-7,-100,False)!=7


def test_taper_and_zero_scale_do_not_create_a_large_deadline_bonus():
    assert potential(500,0,100)==0
    assert abs(potential(500,1/600,100))<1
    assert potential(500,1,0)==0
    assert shaping_reward(0,0,False)==0


def test_moving_closer_receives_better_immediate_feedback():
    start=potential(150,1,100)
    closer=shaping_reward(start,potential(149,599/600,100),False)
    farther=shaping_reward(start,potential(151,599/600,100),False)
    assert closer>farther
    assert potential(15000,1,100)==potential(1505,1,100)


@pytest.mark.parametrize('args',[(float('nan'),1,100),(10,float('inf'),100),(-1,1,100),(10,1,-1),(10,1.1,100)])
def test_invalid_potential_inputs_fail(args):
    with pytest.raises(ValueError):potential(*args)


class DummyWorld(ParallelEnv):
    possible_agents=['A','B']
    metadata={}
    def reset(self,seed=None,options=None):
        self.agents=['A','B'];self.positions={'A':-10.,'B':-20.};self.count=0
        return {a:np.array([0.]) for a in self.agents},{a:{} for a in self.agents}
    def step(self,actions):
        self.actions_received=actions
        active=list(self.agents);self.count+=1
        self.last_observation={a:np.array([self.count]) for a in active}
        self.last_native={a:float(self.count) for a in active}
        self.last_info={a:{'total_reward':float(self.count)} for a in active}
        terminated={a:a=='A' for a in active}
        truncated={a:self.count==2 for a in active}
        self.positions['B']=-8
        self.agents=['B'] if self.count==1 else []
        return self.last_observation,self.last_native,terminated,truncated,self.last_info


class TracedPotential(PotentialProgress):
    def _potential(self,agent):return self.env.positions[agent]


def test_wrapper_preserves_actions_observations_metrics_and_individual_lifetimes():
    base=DummyWorld();wrapped=TracedPotential(base,100)
    wrapped.reset(seed=1)
    commands={'A':np.array([0.,1.]),'B':np.array([1.,0.])}
    obs,rewards,term,trunc,info=wrapped.step(commands)
    assert base.actions_received is commands
    assert obs is base.last_observation and info is base.last_info
    assert base.last_native=={'A':1.,'B':1.}
    assert rewards['A']==11.
    assert rewards['B']==pytest.approx(21-GAMMA*8)
    assert wrapped.potentials=={'B':-8}
    obs,rewards,term,trunc,info=wrapped.step({'B':np.array([0.,0.])})
    assert rewards=={'B':10.} and trunc['B'] and not term['B']
    assert wrapped.potentials=={}
    assert info is base.last_info and info['B']['total_reward']==2.
