from types import SimpleNamespace
import numpy as np
import pytest
from gymnasium import spaces

from atc.route_choice import CLEARANCES_KM, RouteChoiceInput, RouteChoiceResidual, choose_route
from atc.route_input import configuration as input_configuration
from atc.residual import configuration as residual_configuration
from atc.routes import VisibilityPlanner
from atc.submission import _validate_configuration


def planners(sector, obstacles):
    return {c: VisibilityPlanner(sector, obstacles, c, sector_clearance=c) for c in CLEARANCES_KM}


def test_clearance_preference_does_not_force_a_long_obstacle_detour():
    sector = [(-100,-100), (100,-100), (100,100), (-100,100)]
    obstacle = [(-5,-80), (5,-80), (5,95), (-5,95)]
    choices = planners(sector, [obstacle])
    planner, path, length, wider_length = choose_route(choices, [-40,40], [40,40])
    assert planner.clearance == 2
    assert length < wider_length * .6
    assert path[:,1].max() > 95
    assert all(planner.visible(a, [b])[0] for a,b in zip(path[:-1],path[1:]))


def test_wider_clearance_is_retained_for_a_modest_detour():
    sector = [(-100,-100), (100,-100), (100,100), (-100,100)]
    obstacle = [(-5,-5), (5,-5), (5,5), (-5,5)]
    planner, path, length, wider_length = choose_route(planners(sector,[obstacle]),[-50,0],[50,0])
    assert planner.clearance == 6
    assert length == wider_length
    assert np.all(np.isfinite(path))


def test_lower_clearance_is_used_only_when_preferred_paths_are_unavailable():
    sector = [(-20,-20),(20,-20),(20,20),(-20,20)]
    obstacle = [(-2,-5),(2,-5),(2,5),(-2,5)]
    choices=planners(sector,[obstacle])
    chosen=choose_route(choices,[3.5,0],[15,0])
    assert chosen[0].clearance == 1
    assert all(choices[c].path([3.5,0],[15,0]) is None for c in (6,3,2))


def test_disconnected_airspace_has_no_geometric_route():
    sector = [(-20,-20),(20,-20),(20,20),(-20,20)]
    obstacle = [(-2,-25),(2,-25),(2,25),(-2,25)]
    assert choose_route(planners(sector,[obstacle]),[-15,0],[15,0]) is None


class Base:
    def __init__(self):
        self.observation_space=spaces.Dict({'cos_drift':spaces.Box(-1.,1.,(1,)),
                                           'sin_drift':spaces.Box(-1.,1.,(1,))})
    def reset(self):
        return 'reset'
    def _get_action(self, agent, action):
        self.received=action


class ResidualEnv(RouteChoiceResidual, Base):
    pass


def test_route_choice_residual_keeps_zero_reference_dispatch_and_resets():
    env=ResidualEnv()
    env._route_reference=lambda agent, advance=False: None
    env._reference_turns['AC']=np.float32(.25)
    env._get_action('AC',[0,0])
    np.testing.assert_array_equal(env.received,np.array([.25,0],dtype=np.float32))
    env._input_planners[2]=object()
    env.route_input_statistics['route_choices']=5
    env._route_scenario=object()
    assert env.reset() == 'reset'
    assert not env._input_planners and not env._reference_turns
    assert env._route_scenario is None
    assert not any(env.route_input_statistics.values())


def test_route_choice_checkpoint_rejects_stale_selection_settings():
    config=dict(env='ma',algorithm='sac',recipe='public_route_choice_residual',
                **input_configuration(True), **residual_configuration())
    _validate_configuration('ma',config)
    config['route_choice_max_detour_ratio']=2
    with pytest.raises(ValueError,match='route-input settings'):
        _validate_configuration('ma',config)
