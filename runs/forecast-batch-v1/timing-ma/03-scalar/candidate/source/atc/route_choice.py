"""Prefer wider route clearance only when its geometric detour is modest."""
import numpy as np

from atc.route_input import RouteInput, SECTOR_CLEARANCE_KM
from atc.routes import RouteGuidance, VisibilityPlanner
from atc.residual import RouteResidualControl

REVISION = 1
CLEARANCES_KM = (6.0, 3.0, 2.0, 1.0, 0.0)
MIN_PREFERRED_CLEARANCE_KM = 2.0
MAX_DETOUR_RATIO = 1.15


def configuration():
    return dict(route_input_clearances_km=list(CLEARANCES_KM),
                route_choice_revision=REVISION,
                route_choice_min_preferred_clearance_km=MIN_PREFERRED_CLEARANCE_KM,
                route_choice_max_detour_ratio=MAX_DETOUR_RATIO,
                route_choice_fallback='lower_clearance_only_if_no_preferred_path')


def choose_route(planners, start, goal):
    """Return the widest path within 15% of the shortest preferred path.

    Paths at or above the action filter's 2 km margin are preferred. Lower-clearance
    paths are recovery references only when none of those exists; the unchanged
    action filter still determines feasible executed commands.
    """
    candidates = []
    for clearance in sorted(planners, reverse=True):
        planner = planners[clearance]
        path = planner.path(start, goal)
        if path is not None:
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            candidates.append((planner, path, length))
    if not candidates:
        return None
    preferred = [entry for entry in candidates if entry[0].clearance >= MIN_PREFERRED_CLEARANCE_KM]
    pool = preferred or candidates
    shortest = min(entry[2] for entry in pool)
    chosen = next(entry for entry in pool if entry[2] <= shortest * MAX_DETOUR_RATIO + 1e-9)
    return (*chosen, pool[0][2])


class RouteChoiceInput(RouteInput):
    def _clear_input_state(self):
        super()._clear_input_state()
        self.route_input_statistics.update(route_choices=0, shorter_route_choices=0,
            lower_clearance_fallbacks=0, geometric_detour_avoided_km=0.0,
            unroutable_initializations=0)

    def _choose_route(self, start, goal):
        chosen = choose_route(self._input_planners, start, goal)
        if chosen is not None:
            planner, path, length, widest_length = chosen
            stats = self.route_input_statistics
            stats['route_choices'] += 1
            stats['shorter_route_choices'] += int(widest_length - length > 1e-6)
            stats['geometric_detour_avoided_km'] += max(0.0, widest_length - length)
            stats['lower_clearance_fallbacks'] += int(planner.clearance < MIN_PREFERRED_CLEARANCE_KM)
            return planner, path
        return None

    def _ensure_routes(self):
        if self._route_scenario is self.scenario:
            return
        self._route_scenario = self.scenario
        self._input_planners = {clearance: VisibilityPlanner(
            self._xy(self.scenario.sector),
            [self._xy(obstacle.vertices) for obstacle in self.scenario.obstacles], clearance,
            sector_clearance=min(clearance, SECTOR_CLEARANCE_KM)) for clearance in CLEARANCES_KM}
        self._routes, self._planners, self._route_indices = {}, {}, {}
        for spec in self.scenario.agents:
            start, goal = self._xy(spec.start), self._xy(spec.goal)
            chosen = self._choose_route(start, goal)
            if chosen is None:
                planner, path = self._input_planners[0.0], np.stack((start, goal))
                self.route_input_statistics['unroutable_initializations'] += 1
            else:
                planner, path = chosen
            self._routes[spec.ac_id] = path
            self._planners[spec.ac_id] = planner
            self._route_indices[spec.ac_id] = 1

    def _route_reference(self, ac_id, advance=False):
        import bluesky as bs
        self._ensure_routes()
        index = bs.traf.id2idx(ac_id)
        current = self._xy((bs.traf.lat[index], bs.traf.lon[index]))
        path = self._routes[ac_id]
        visible = self._planners[ac_id].visible(current, path[self._route_indices[ac_id]:]).any()
        decision_time = float(bs.sim.simt)
        if not visible and self._input_replan_clock.get(ac_id) != decision_time:
            self._input_replan_clock[ac_id] = decision_time
            chosen = self._choose_route(current, path[-1])
            if chosen is None:
                self.route_input_statistics['failed_replans'] += 1
            else:
                self._planners[ac_id], self._routes[ac_id] = chosen
                self._route_indices[ac_id] = 1
                self.route_input_statistics['route_replans'] += 1
        return RouteGuidance._route_reference(self, ac_id, advance=advance)


class RouteChoiceResidual(RouteResidualControl, RouteChoiceInput):
    """Keep the learned residual mapping while changing geometric route selection."""
