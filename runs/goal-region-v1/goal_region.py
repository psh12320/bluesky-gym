"""Experimental guidance to a reachable region around the original waypoint."""
import heapq
import numpy as np
from shapely import covers, points
from shapely.geometry import Point
from shapely.ops import nearest_points
from atc.routes import RouteGuidance
from atc.route_input import navigation_observation
from atc.route_choice import RouteChoiceInput, MIN_PREFERRED_CLEARANCE_KM, MAX_DETOUR_RATIO
from atc.residual import RouteResidualControl

NAVIGATION_RADIUS_KM = 4.75
ENDPOINT_DISTANCE_LIMIT_KM = 4.95


def inverse_xy(coordinates, center):
    coordinates = np.asarray(coordinates, dtype=float)
    cosine = np.cos(np.deg2rad(center[0]))
    if abs(cosine) < 1e-8:
        raise ValueError('Local planar navigation is undefined at the pole')
    return np.stack((center[0] + coordinates[..., 1] / (60 * 1.852),
                     center[1] + coordinates[..., 0] / (60 * 1.852 * cosine)), axis=-1)


def valid_scored_endpoints(coordinates, goal, center, limit=ENDPOINT_DISTANCE_LIMIT_KM):
    from core.tools import kwikqdrdist
    targets = inverse_xy(coordinates, center)
    original = inverse_xy(goal, center)
    _, distance = kwikqdrdist(targets[..., 0], targets[..., 1], original[0], original[1])
    return np.asarray(distance * 1.852 < limit, dtype=bool)


def region_endpoints(planner, start, goal, radius):
    """Candidate endpoints: centre, radial projections and circle/edge crossings."""
    nodes = np.vstack((start, planner.vertices))
    displacement = nodes - goal
    norms = np.linalg.norm(displacement, axis=1)
    radial = goal + displacement * np.minimum(1, radius / np.maximum(norms, 1e-15))[:, None]
    candidates = [np.asarray(goal).reshape(1, 2), radial]
    if not planner.domain.is_empty:
        nearest = nearest_points(planner.domain, Point(goal))[0]
        candidates.append(np.array([[nearest.x, nearest.y]]))
    polygons = [planner.domain] if planner.domain.geom_type == 'Polygon' else list(planner.domain.geoms)
    for polygon in polygons:
        if polygon.geom_type != 'Polygon':
            continue
        for ring in [polygon.exterior, *polygon.interiors]:
            coordinates = np.asarray(ring.coords)
            beginnings, vectors = coordinates[:-1], np.diff(coordinates, axis=0)
            relative = beginnings - goal
            a = np.sum(vectors * vectors, axis=1)
            b = 2 * np.sum(relative * vectors, axis=1)
            c = np.sum(relative * relative, axis=1) - radius * radius
            discriminant = b * b - 4 * a * c
            valid = (a > 1e-15) & (discriminant >= 0)
            for sign in (-1, 1):
                fraction = (-b + sign * np.sqrt(np.maximum(discriminant, 0))) / np.where(a > 1e-15, 2 * a, 1)
                selected = valid & (fraction >= 0) & (fraction <= 1)
                candidates.append(beginnings[selected] + fraction[selected, None] * vectors[selected])
    result = np.unique(np.vstack(candidates), axis=0)
    allowed = np.linalg.norm(result - goal, axis=1) <= radius + 1e-8
    allowed &= np.asarray(covers(planner.visible_domain, points(result)), dtype=bool)
    return result[allowed]


def path_to_region(planner, start, goal, radius=NAVIGATION_RADIUS_KM, endpoint_valid=None):
    """Shortest visibility-graph path to a finite set inside the navigation disk.

    This is a geometric approximation; it does not guarantee dynamic feasibility
    or optimality over every point in the continuous capture region.
    """
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if start.shape != (2,) or goal.shape != (2,) or not np.isfinite([*start, *goal, radius]).all() or radius <= 0:
        raise ValueError('Finite two-dimensional points and a positive radius are required')
    if planner.domain.is_empty or not planner.visible_domain.covers(Point(start)):
        return None
    endpoints = region_endpoints(planner, start, goal, radius)
    if endpoint_valid is not None:
        endpoints = endpoints[np.asarray(endpoint_valid(endpoints), dtype=bool)]
    if not len(endpoints):
        return None
    count = len(planner.vertices)
    costs, parents = np.full(count, np.inf), np.full(count, -1, dtype=int)
    queue = []
    for i in np.flatnonzero(planner.visible(start, planner.vertices)):
        costs[i] = np.linalg.norm(planner.vertices[i] - start)
        heapq.heappush(queue, (costs[i], int(i)))
    while queue:
        cost, i = heapq.heappop(queue)
        if cost != costs[i]:
            continue
        for j, distance in planner.edges[i]:
            candidate = cost + distance
            if candidate < costs[j]:
                costs[j], parents[j] = candidate, i
                heapq.heappush(queue, (candidate, j))
    best_cost, best_last, best_endpoint = np.inf, -1, None
    nodes = [(-1, start, 0.0)] + [(int(i), planner.vertices[i], float(costs[i])) for i in np.argsort(costs) if np.isfinite(costs[i])]
    for index, node, cost in nodes:
        if cost >= best_cost:
            continue
        visible = np.flatnonzero(planner.visible(node, endpoints))
        if not len(visible):
            continue
        terminal_costs = np.linalg.norm(endpoints[visible] - node, axis=1)
        j = int(np.argmin(terminal_costs))
        if cost + terminal_costs[j] < best_cost:
            best_cost, best_last, best_endpoint = cost + terminal_costs[j], index, endpoints[visible[j]]
    if best_endpoint is None:
        return None
    indices = []
    while best_last >= 0:
        indices.append(best_last)
        best_last = int(parents[best_last])
    return np.vstack((start, planner.vertices[indices[::-1]], best_endpoint))


def choose_region_route(planners, start, goal, endpoint_valid=None):
    candidates = []
    for clearance in sorted(planners, reverse=True):
        planner = planners[clearance]
        path = path_to_region(planner, start, goal, endpoint_valid=endpoint_valid)
        if path is not None:
            candidates.append((planner, path, float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())))
    if not candidates:
        return None
    preferred = [item for item in candidates if item[0].clearance >= MIN_PREFERRED_CLEARANCE_KM]
    pool = preferred or candidates
    shortest = min(item[2] for item in pool)
    chosen = next(item for item in pool if item[2] <= shortest * MAX_DETOUR_RATIO + 1e-9)
    return (*chosen, pool[0][2])


class GoalRegionInput(RouteChoiceInput):
    def _ensure_routes(self):
        if self._route_scenario is not self.scenario:
            self._region_original_goals = {spec.ac_id: self._xy(spec.goal) for spec in self.scenario.agents}
        return super()._ensure_routes()

    def _choose_route(self, start, goal):
        valid = lambda endpoints: valid_scored_endpoints(endpoints, goal, self.scenario.center,
                                                         min(ENDPOINT_DISTANCE_LIMIT_KM, self.distance_margin - .05))
        chosen = choose_region_route(self._input_planners, start, goal, endpoint_valid=valid)
        if chosen is None:
            return None
        planner, path, length, widest_length = chosen
        stats = self.route_input_statistics
        stats['route_choices'] += 1
        stats['shorter_route_choices'] += int(widest_length - length > 1e-6)
        stats['geometric_detour_avoided_km'] += max(0.0, widest_length - length)
        stats['lower_clearance_fallbacks'] += int(planner.clearance < MIN_PREFERRED_CLEARANCE_KM)
        stats['region_routes'] = stats.get('region_routes', 0) + 1
        assert np.linalg.norm(path[-1] - goal) <= NAVIGATION_RADIUS_KM + 1e-8
        assert valid(path[-1:])[0]
        return planner, path

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
            # Replan around the assigned waypoint, never around a previous endpoint.
            chosen = self._choose_route(current, self._region_original_goals[ac_id])
            if chosen is None:
                self.route_input_statistics['failed_replans'] += 1
            else:
                self._planners[ac_id], self._routes[ac_id] = chosen
                self._route_indices[ac_id] = 1
                self.route_input_statistics['route_replans'] += 1
        return RouteGuidance._route_reference(self, ac_id, advance=advance)

    def _get_obs(self, ac_id):
        import bluesky as bs
        observation = RouteGuidance._get_obs(self, ac_id)
        index = bs.traf.id2idx(ac_id)
        current = self._xy((bs.traf.lat[index], bs.traf.lon[index]))
        visible = bool(self._planners[ac_id].visible(current, [self._region_original_goals[ac_id]])[0])
        self.route_input_statistics['observations'] += 1
        self.route_input_statistics['redirected_observations'] += int(not visible)
        return navigation_observation(observation, visible)


class GoalRegionResidual(RouteResidualControl, GoalRegionInput):
    pass


def install():
    """Change only route input classes in this experimental process."""
    import atc.route_choice as choice
    assert choice.RouteChoiceInput is RouteChoiceInput, 'Install once in a fresh process'
    choice.RouteChoiceInput, choice.RouteChoiceResidual = GoalRegionInput, GoalRegionResidual
