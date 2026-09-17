from copy import deepcopy
import pytest
from intervention_reward import correction_cost, InterventionReward

def record(nominal_h=0,selected_h=0,nominal_v=1,selected_v=1,intervened=True):
    return dict(reference_heading_turn_deg=nominal_h,nominal_heading_adjustment_deg=0,
        commanded_heading_turn_deg=selected_h,nominal_speed_action=nominal_v,
        commanded_speed_action=selected_v,traffic_projection_intervened=intervened)

def test_scale_and_extremes():
    assert correction_cost(record(-45,45,-1,1))==1
    assert correction_cost(record(0,45,1,1))==pytest.approx(.125)
    assert correction_cost(record(0,0,1,-1))==pytest.approx(.5)
    assert correction_cost(record(intervened=False))==0

def test_residual_reconstructs_nominal_command():
    r=record(20,35)
    r['nominal_heading_adjustment_deg']=15
    assert correction_cost(r)==0

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),46])
def test_invalid_commands_fail(bad):
    with pytest.raises(ValueError): correction_cost(record(selected_h=bad))

class World:
    metadata={}
    possible_agents=['A','B']
    def __init__(self):
        self.last_projection={}
        self.calls=0
        self.agents=['A','B']
        self.payload=({'A':[1],'B':[2]},{'A':3.0,'B':4.0},{'A':True,'B':False},
                      {'A':False,'B':False},{'A':{'total_reward':3.0},'B':{'total_reward':4.0}})
    @property
    def unwrapped(self):return self
    def step(self,actions):
        self.calls+=1
        self.last_projection={'A':record(-45,45,-1,1),'B':record(intervened=False)}
        return self.payload

def test_terminal_reward_and_scoring_are_separate():
    world=World(); expected=deepcopy(world.payload)
    env=InterventionReward(world,1)
    result=env.step({'A':[0,0],'B':[0,0]})
    assert result[1]=={'A':2.0,'B':4.0}
    assert world.payload==expected and world.calls==1
    for i in (0,2,3,4):assert result[i] is world.payload[i]
    assert env.statistics['commands']==2 and env.statistics['penalty']==1

def test_zero_coefficient_preserves_rewards():
    world=World();env=InterventionReward(world,0)
    assert env.step({'A':[0,0],'B':[0,0]})==world.payload
    assert env.statistics['correction_cost']==1 and env.statistics['penalty']==0

def test_stale_diagnostics_are_rejected():
    world=World();env=InterventionReward(world,1)
    env.step({'A':[0,0],'B':[0,0]})
    world.step=lambda actions:world.payload
    with pytest.raises(ValueError,match='stale'):env.step({'A':[0,0],'B':[0,0]})

@pytest.mark.parametrize('coefficient',[-1,float('nan'),float('inf')])
def test_invalid_coefficient(coefficient):
    with pytest.raises(ValueError):InterventionReward(World(),coefficient)
