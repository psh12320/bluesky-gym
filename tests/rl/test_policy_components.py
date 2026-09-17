import numpy as np
import pytest
from atc_rl.policy_components import combine_policy_components


@pytest.mark.parametrize("batch",[False,True])
def test_component_swaps_reconstruct_both_controls_and_do_not_mutate_inputs(batch):
    initial=np.array([[-1.,.123456789],[.75,1.]],dtype=np.float64)
    trained=np.array([[.3,-1.],[1.,-.8]],dtype=np.float64)
    if not batch:initial,trained=initial[0],trained[0]
    before=(initial.copy(),trained.copy())
    a=combine_policy_components(initial,trained,"trained","initial")
    b=combine_policy_components(initial,trained,"initial","trained")
    np.testing.assert_array_equal(a[...,0],trained[...,0].astype(np.float32))
    np.testing.assert_array_equal(a[...,1],initial[...,1].astype(np.float32))
    np.testing.assert_array_equal(b[...,0],initial[...,0].astype(np.float32))
    np.testing.assert_array_equal(b[...,1],trained[...,1].astype(np.float32))
    np.testing.assert_array_equal(combine_policy_components(initial,trained,"initial","initial"),initial.astype(np.float32))
    np.testing.assert_array_equal(combine_policy_components(initial,trained,"trained","trained"),trained.astype(np.float32))
    assert a.dtype==b.dtype==np.float32
    np.testing.assert_array_equal(initial,before[0]);np.testing.assert_array_equal(trained,before[1])


@pytest.mark.parametrize("initial,trained",[
    ([0,0],[0]),([[0,0]],[0,0]),([0,float("nan")],[0,0]),([0,0],[0,float("inf")]),([1.1,0],[0,0])])
def test_invalid_commands_are_rejected_before_entering_the_simulator(initial,trained):
    with pytest.raises(ValueError):combine_policy_components(initial,trained,"initial","trained")


def test_invalid_source_label_cannot_silently_select_another_policy():
    with pytest.raises(ValueError):combine_policy_components([0,0],[1,1],"nominal","trained")
