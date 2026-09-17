"""Obstacle-route bearing input for existing direct-control policies."""
from gymnasium import spaces

from atc.routes import ROUTE_REVISION, RouteGuidance, VisibilityPlanner

REVISION = 1
CLEARANCES_KM = (6.0, 3.0, 1.0, 0.0)
SECTOR_CLEARANCE_KM = 6.0
ROUTE_FIELDS = ('route_cos_drift', 'route_sin_drift', 'route_target_distance')


def configuration(route_choice=False):
    result = dict(route_input_revision=REVISION, route_input_planner_revision=ROUTE_REVISION,
                route_input_clearances_km=list(CLEARANCES_KM),
                route_input_sector_clearance_km=SECTOR_CLEARANCE_KM,
                route_input_distance='original_goal', route_input_direct_bearing='original_observation')
    if route_choice:
        from atc.route_choice import configuration as choice_configuration
        result.update(choice_configuration())
    return result


def navigation_observation(observation, direct_goal_visible):
    """Reuse the original vector layout; redirect only the goal-bearing channels."""
    result = {key: value for key, value in observation.items() if key not in ROUTE_FIELDS}
    if not direct_goal_visible:
        result['cos_drift'] = observation['route_cos_drift']
        result['sin_drift'] = observation['route_sin_drift']
    return result


class RouteInput(RouteGuidance):
    """Supply a static-route bearing while retaining direct heading/speed actions.

    This reuses the existing polygonal planner and route progress bookkeeping.
    Candidate-action correction and all actual scoring remain separate.
    """

    def __init__(self, *args, **kwargs):
        self._clear_input_state()
        super().__init__(*args, route_sector_clearance=SECTOR_CLEARANCE_KM,
                         guard_static=False, **kwargs)
        def original_layout(space):
            return spaces.Dict({key: value for key, value in space.spaces.items() if key not in ROUTE_FIELDS})
        if hasattr(self, 'observation_spaces'):
            self.observation_spaces = {agent: original_layout(space) for agent, space in self.observation_spaces.items()}
        else:
            self.observation_space = original_layout(self.observation_space)

    def _clear_input_state(self):
        self._input_planners = {}
        self._input_replan_clock = {}
        self.route_input_statistics = dict(observations=0, redirected_observations=0,
                                          route_replans=0, failed_replans=0)

    def reset(self, *args, **kwargs):
        self._clear_input_state()
        return super().reset(*args, **kwargs)

    def _route_reference(self, ac_id, advance=False):
        import bluesky as bs
        self._ensure_routes()
        index = bs.traf.id2idx(ac_id)
        current = self._xy((bs.traf.lat[index], bs.traf.lon[index]))
        path = self._routes[ac_id]
        remaining = path[self._route_indices[ac_id]:]
        visible = self._planners[ac_id].visible(current, remaining).any()
        decision_time = float(bs.sim.simt)
        if not visible and self._input_replan_clock.get(ac_id) != decision_time:
            self._input_replan_clock[ac_id] = decision_time
            self._input_planners.update({planner.clearance: planner for planner in self._planners.values()})
            for clearance in CLEARANCES_KM:
                if clearance not in self._input_planners:
                    self._input_planners[clearance] = VisibilityPlanner(
                        self._xy(self.scenario.sector),
                        [self._xy(obstacle.vertices) for obstacle in self.scenario.obstacles], clearance,
                        sector_clearance=min(clearance, SECTOR_CLEARANCE_KM))
                planner = self._input_planners[clearance]
                replacement = planner.path(current, path[-1])
                if replacement is not None:
                    self._routes[ac_id] = replacement
                    self._planners[ac_id] = planner
                    self._route_indices[ac_id] = 1
                    self.route_input_statistics['route_replans'] += 1
                    break
            else:
                self.route_input_statistics['failed_replans'] += 1
        return super()._route_reference(ac_id, advance=advance)

    def _get_obs(self, ac_id):
        import bluesky as bs
        observation = super()._get_obs(ac_id)
        index = bs.traf.id2idx(ac_id)
        current = self._xy((bs.traf.lat[index], bs.traf.lon[index]))
        visible = bool(self._planners[ac_id].visible(current, [self._routes[ac_id][-1]])[0])
        self.route_input_statistics['observations'] += 1
        self.route_input_statistics['redirected_observations'] += int(not visible)
        return navigation_observation(observation, visible)

    def _get_action(self, ac_id, action):
        self._route_reference(ac_id, advance=True)
        # Skip RouteGuidance's route-relative action mapping. The policy output
        # remains a direct command and reaches the optional joint correction next.
        return super(RouteGuidance, self)._get_action(ac_id, action)
