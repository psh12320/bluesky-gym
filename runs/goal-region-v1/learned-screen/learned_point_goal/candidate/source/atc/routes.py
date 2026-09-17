import heapq

import numpy as np
from shapely import covers, linestrings
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROUTE_REVISION = 2


class VisibilityPlanner:
    """Shortest polygonal paths through a sector with buffered restricted areas.

    Coordinates are local east/north kilometres. This plans geometric paths only;
    flight dynamics and separation from moving aircraft remain controller concerns.
    """

    def __init__(self, sector, obstacles, clearance=6.0, sector_clearance=0.0):
        self.clearance = clearance
        self.sector = Polygon(sector)
        blocked = unary_union([Polygon(p).buffer(clearance, join_style=2) for p in obstacles])
        self.domain = self.sector.buffer(-sector_clearance).difference(blocked)
        self.visible_domain = self.domain.buffer(1e-7)
        polygons = [self.domain] if self.domain.geom_type == "Polygon" else list(self.domain.geoms)
        points = []
        for polygon in polygons:
            if polygon.geom_type != "Polygon":
                continue
            points.extend(polygon.exterior.coords[:-1])
            for ring in polygon.interiors:
                points.extend(ring.coords[:-1])
        self.vertices = np.asarray(points, dtype=float).reshape(-1, 2)
        self.edges = []
        for i, point in enumerate(self.vertices):
            visible = self.visible(point, self.vertices)
            self.edges.append([(int(j), float(np.linalg.norm(point - self.vertices[j])))
                               for j in np.flatnonzero(visible) if j != i])

    def visible(self, start, targets):
        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        if not len(targets):
            return np.zeros(0, dtype=bool)
        pairs = np.stack((np.broadcast_to(start, targets.shape), targets), axis=1)
        return np.asarray(covers(self.visible_domain, linestrings(pairs)), dtype=bool)

    def path(self, start, goal):
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        if self.visible(start, [goal])[0]:
            return np.stack((start, goal))
        count = len(self.vertices)
        from_start, to_goal = self.visible(start, self.vertices), self.visible(goal, self.vertices)
        costs = np.full(count, np.inf)
        parent = np.full(count, -1, dtype=int)
        queue = []
        for i in np.flatnonzero(from_start):
            costs[i] = np.linalg.norm(self.vertices[i] - start)
            heapq.heappush(queue, (costs[i], int(i)))
        best_cost, best_last = np.inf, -1
        while queue:
            distance, i = heapq.heappop(queue)
            if distance != costs[i] or distance >= best_cost:
                continue
            if to_goal[i]:
                candidate = distance + np.linalg.norm(self.vertices[i] - goal)
                if candidate < best_cost:
                    best_cost, best_last = candidate, i
            for j, length in self.edges[i]:
                candidate = distance + length
                if candidate < costs[j]:
                    costs[j], parent[j] = candidate, i
                    heapq.heappush(queue, (candidate, j))
        if best_last < 0:
            return None
        indices = []
        while best_last >= 0:
            indices.append(best_last)
            best_last = parent[best_last]
        return np.vstack((start, self.vertices[indices[::-1]], goal))

    def target_index(self, current, path, minimum_index=1):
        """Advance visible waypoints without reacquiring a previously passed one."""
        minimum_index = min(max(int(minimum_index), 1), len(path) - 1)
        visible = self.visible(current, path[minimum_index:])
        if visible.any():
            return minimum_index + int(np.flatnonzero(visible)[-1])
        return minimum_index

    def target(self, current, path, minimum_index=1):
        return path[self.target_index(current, path, minimum_index)]


class RouteGuidance:
    """Expose a geometric route reference and learn heading offsets around it."""

    def __init__(self, *args, route_sector_clearance=0.0, guard_static=False, **kwargs):
        self._route_sector_clearance = route_sector_clearance
        self._guard_static = guard_static
        super().__init__(*args, **kwargs)
        from gymnasium import spaces
        extra = {
            "route_cos_drift": spaces.Box(-1.0, 1.0, (1,), dtype=np.float64),
            "route_sin_drift": spaces.Box(-1.0, 1.0, (1,), dtype=np.float64),
            "route_target_distance": spaces.Box(0.0, np.inf, (1,), dtype=np.float64),
        }
        if hasattr(self, "observation_spaces"):
            self.observation_spaces = {a: spaces.Dict({**s.spaces, **extra})
                                       for a, s in self.observation_spaces.items()}
        else:
            self.observation_space = spaces.Dict({**self.observation_space.spaces, **extra})
        self._route_scenario = None

    def reset(self, *args, **kwargs):
        self._route_scenario = None
        return super().reset(*args, **kwargs)

    def _xy(self, points):
        points = np.asarray(points, dtype=float)
        north = (points[..., 0] - self.scenario.center[0]) * 60 * 1.852
        east = (points[..., 1] - self.scenario.center[1]) * 60 * 1.852 * np.cos(np.deg2rad(self.scenario.center[0]))
        return np.stack((east, north), axis=-1)

    def _ensure_routes(self):
        if self._route_scenario is self.scenario:
            return
        self._route_scenario = self.scenario
        self._routes, self._planners, self._route_indices = {}, {}, {}
        if self._guard_static:
            blocked = unary_union([Polygon(self._xy(o.vertices)).buffer(2.0, join_style=2)
                                   for o in self.scenario.obstacles])
            self._guard_domain = Polygon(self._xy(self.scenario.sector)).buffer(-2.0).difference(blocked)
        remaining = list(self.scenario.agents)
        for clearance in (6.0, 3.0, 1.0, 0.0):
            planner = VisibilityPlanner(self._xy(self.scenario.sector),
                                        [self._xy(o.vertices) for o in self.scenario.obstacles], clearance,
                                        sector_clearance=min(clearance, self._route_sector_clearance))
            unresolved = []
            for agent in remaining:
                path = planner.path(self._xy(agent.start), self._xy(agent.goal))
                if path is None:
                    unresolved.append(agent)
                else:
                    self._routes[agent.ac_id] = path
                    self._planners[agent.ac_id] = planner
            remaining = unresolved
            if not remaining:
                break
        self._route_indices = {agent.ac_id: 1 for agent in self.scenario.agents}
        for agent in remaining:
            self._routes[agent.ac_id] = np.stack((self._xy(agent.start), self._xy(agent.goal)))
            self._planners[agent.ac_id] = planner

    def _route_reference(self, ac_id, advance=False):
        import bluesky as bs
        self._ensure_routes()
        idx = bs.traf.id2idx(ac_id)
        current = self._xy((bs.traf.lat[idx], bs.traf.lon[idx]))
        index = self._planners[ac_id].target_index(current, self._routes[ac_id], self._route_indices[ac_id])
        if advance:
            self._route_indices[ac_id] = index
        target = self._routes[ac_id][index]
        delta = target - current
        bearing = float(np.rad2deg(np.arctan2(delta[0], delta[1])))
        return bearing, float(np.linalg.norm(delta))

    def _get_obs(self, ac_id):
        import bluesky as bs
        result = super()._get_obs(ac_id)
        bearing, distance = self._route_reference(ac_id)
        drift = np.deg2rad(bs.traf.hdg[bs.traf.id2idx(ac_id)] - bearing)
        result.update(route_cos_drift=np.array([np.cos(drift)]),
                      route_sin_drift=np.array([np.sin(drift)]),
                      route_target_distance=np.array([distance / 170.0]))
        return result

    def _get_action(self, ac_id, action):
        import bluesky as bs
        from atc.control import goal_relative_command
        bearing, _ = self._route_reference(ac_id, advance=True)
        heading = bs.traf.hdg[bs.traf.id2idx(ac_id)]
        offset = float(action[0]) * 45.0
        if self._guard_static:
            from atc.static_filter import filter_heading
            idx = bs.traf.id2idx(ac_id)
            position = self._xy((bs.traf.lat[idx], bs.traf.lon[idx]))
            offset = filter_heading(self._guard_domain, position, heading, bearing, offset,
                                    max(float(bs.traf.tas[idx]), 150.0), self.d_heading)
        command = goal_relative_command(offset / 45.0, heading, bearing, self.d_heading, offset_limit=45.0)
        self.heading_action.execute(ac_id, command)
        self.speed_action.execute(ac_id, action[1])
