import sys
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from atc.route_input import RouteInput, ROUTE_FIELDS, configuration, navigation_observation
from atc.routes import VisibilityPlanner
from atc.submission import _validate_configuration


def test_route_input_changes_only_bearing_and_keeps_original_goal_distance():
    observation = dict(cos_drift=np.array([.1]), sin_drift=np.array([.2]),
        distance=np.array([.9]), traffic=np.arange(9), route_cos_drift=np.array([.3]),
        route_sin_drift=np.array([.4]), route_target_distance=np.array([.01]))
    direct = navigation_observation(observation, True)
    redirected = navigation_observation(observation, False)
    assert set(direct) == set(redirected) == set(observation) - set(ROUTE_FIELDS)
    for key in direct:
        assert direct[key] is observation[key]
    for key in ('distance', 'traffic'):
        assert redirected[key] is observation[key]
    assert redirected['cos_drift'] is observation['route_cos_drift']
    assert redirected['sin_drift'] is observation['route_sin_drift']
    assert len(observation) == 7


class DirectBase:
    def __init__(self):
        self.observation_space = spaces.Dict({'cos_drift': spaces.Box(-1., 1., (1,)),
                                              'distance': spaces.Box(0., 10., (1,))})
    def reset(self):
        return 'reset'
    def _get_action(self, ac_id, action):
        self.received = (ac_id, action)
        return 'direct'


class InputEnv(RouteInput, DirectBase):
    pass


def test_action_dispatch_preserves_direct_command_and_original_space():
    env = InputEnv()
    assert env.observation_space == DirectBase().observation_space
    calls = []
    env._route_reference = lambda ac_id, advance=False: calls.append((ac_id, advance))
    command = np.array([.123456789, -.7])
    assert env._get_action('AC', command) == 'direct'
    assert env.received[1] is command
    assert calls == [('AC', True)]


def test_reset_discards_route_replan_state():
    env = InputEnv()
    env._input_planners[1] = object()
    env._input_replan_clock['AC'] = 10
    env.route_input_statistics['failed_replans'] = 2
    env._route_scenario = object()
    assert env.reset() == 'reset'
    assert env._route_scenario is None
    assert env._input_planners == env._input_replan_clock == {}
    assert not any(env.route_input_statistics.values())


def test_off_route_deviation_replans_from_current_position(monkeypatch):
    sector = [(-40, -40), (40, -40), (40, 40), (-40, 40)]
    obstacle = [(-2, -5), (2, -5), (2, 5), (-2, 5)]
    planner = VisibilityPlanner(sector, [obstacle], 6, sector_clearance=6)
    original = planner.path([-20, 0], [20, 0])
    env = InputEnv()
    env.scenario = SimpleNamespace(sector=sector, obstacles=[SimpleNamespace(vertices=obstacle)])
    env._xy = lambda coordinates: np.asarray(coordinates)
    env._ensure_routes = lambda: None
    env._routes = {'AC': original}
    env._planners = {'AC': planner}
    env._route_indices = {'AC': len(original) - 1}
    traffic = SimpleNamespace(id2idx=lambda ac_id: 0, lat=[-20], lon=[0], hdg=[90])
    monkeypatch.setitem(sys.modules, 'bluesky', SimpleNamespace(traf=traffic, sim=SimpleNamespace(simt=10)))
    bearing, distance = env._route_reference('AC', advance=True)
    assert env.route_input_statistics['route_replans'] == 1
    assert env.route_input_statistics['failed_replans'] == 0
    assert env._route_indices['AC'] < len(original) - 1
    assert np.isfinite([bearing, distance]).all()
    assert env._planners['AC'].visible([-20, 0], [env._routes['AC'][env._route_indices['AC']]])[0]


def test_unreachable_route_reports_failure_once_per_decision(monkeypatch):
    sector = [(-20, -20), (20, -20), (20, 20), (-20, 20)]
    obstacle = [(-2, -25), (2, -25), (2, 25), (-2, 25)]
    env = InputEnv()
    env.scenario = SimpleNamespace(sector=sector, obstacles=[SimpleNamespace(vertices=obstacle)])
    env._xy = lambda coordinates: np.asarray(coordinates)
    env._ensure_routes = lambda: None
    env._routes = {'AC': np.array([[-15, 0], [15, 0]])}
    env._planners = {'AC': VisibilityPlanner(sector, [obstacle], 0)}
    env._route_indices = {'AC': 1}
    sim = SimpleNamespace(simt=10)
    monkeypatch.setitem(sys.modules, 'bluesky', SimpleNamespace(sim=sim,
        traf=SimpleNamespace(id2idx=lambda ac_id: 0, lat=[-15], lon=[0], hdg=[90])))
    env._route_reference('AC')
    env._route_reference('AC', advance=True)
    assert env.route_input_statistics['failed_replans'] == 1
    sim.simt = 20
    env._route_reference('AC')
    assert env.route_input_statistics['failed_replans'] == 2


def test_checkpoint_rejects_changed_route_input_definition():
    config = dict(env='ma', algorithm='sac', recipe='public_route_input', **configuration())
    _validate_configuration('ma', config)
    config['route_input_distance'] = 'route_target'
    with pytest.raises(ValueError, match='route-input settings'):
        _validate_configuration('ma', config)
