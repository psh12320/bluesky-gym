import numpy as np
import pytest
from atc_rl.world_worker import pack_observations

def test_joint_state_masks_completed_aircraft_and_identifies_value_target():
    agents=['A','B','C']
    obs={'A':np.array([1,2]),'B':np.array([9,9]),'C':np.array([3,4])}
    result=pack_observations(obs,['A','C'],agents,2,.5)
    assert result['actor'].tolist()==[[1,2,.5],[0,0,0],[3,4,.5]]
    assert result['critic'][0,:9].tolist()==[1,2,.5,0,0,0,3,4,.5]
    assert result['critic'][0,9:12].tolist()==[1,0,1]
    assert np.array_equal(result['critic'][:,-3:],np.eye(3))
    assert np.array_equal(result['critic'][0,:-3],result['critic'][1,:-3])

def test_other_aircraft_changes_only_context_of_unchanged_actor():
    first=pack_observations({'A':[1,2],'B':[3,4]},['A','B'],['A','B'],2,1)
    second=pack_observations({'A':[1,2],'B':[7,8]},['A','B'],['A','B'],2,1)
    assert np.array_equal(first['actor'][0],second['actor'][0])
    assert not np.array_equal(first['critic'][0],second['critic'][0])

@pytest.mark.parametrize('remaining',[-.1,1.1])
def test_time_fraction_validation(remaining):
    with pytest.raises(ValueError):pack_observations({'A':[0]},['A'],['A'],1,remaining)
