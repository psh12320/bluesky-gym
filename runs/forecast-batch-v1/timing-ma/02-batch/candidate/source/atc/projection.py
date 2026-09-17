"""Project direct heading commands away from predicted static violations."""

import numpy as np
from shapely import covers, difference, distance, length, linestrings, points
from shapely.geometry import Polygon
from shapely.ops import unary_union

from atc.static_filter import predicted_paths

PROJECTION_REVISION = 4
HORIZON_SECONDS = 90.0
CLEARANCE_KM = 2.0


def stop_at_goal(paths, goal, radius):
    """Stop each predicted polyline at its first entry into the goal circle."""
    paths = np.asarray(paths, dtype=float).copy()
    starts, segments = paths[:, :-1], np.diff(paths, axis=1)
    relative = starts - np.asarray(goal, dtype=float)
    a = np.sum(segments * segments, axis=-1)
    b = 2 * np.sum(relative * segments, axis=-1)
    c = np.sum(relative * relative, axis=-1) - radius * radius
    discriminant = b * b - 4 * a * c
    roots = (-b - np.sqrt(np.maximum(discriminant, 0))) / np.where(a > 1e-12, 2 * a, 1)
    inside = c < 0
    entries = inside | ((a > 1e-12) & (discriminant > 0) & (roots >= 0) & (roots < 1))
    fractions = np.where(inside, 0.0, roots)
    for row in np.flatnonzero(entries.any(axis=1)):
        segment = int(np.argmax(entries[row]))
        entry = starts[row, segment] + fractions[row, segment] * segments[row, segment]
        paths[row, segment + 1:] = entry
    return paths


def project_turn(domain, position, heading, nominal_turn, speed_mps, goal_bearing,
                 goal_position=None, goal_radius=5.0):
    """Return (turn, changed, no_feasible_candidate) under an approximate predictor.

    Speed is unchanged. The predictor assumes constant speed and a bounded turn
    toward each command; it is not the actual BlueSky dynamics or a safety proof.
    """
    nominal_turn = float(np.clip(nominal_turn, -45.0, 45.0))
    nominal_path = predicted_paths(position, heading, [heading + nominal_turn], speed_mps,
                                   seconds=HORIZON_SECONDS)
    if goal_position is not None:
        nominal_path = stop_at_goal(nominal_path, goal_position, goal_radius)
    if bool(covers(domain, linestrings(nominal_path))[0]):
        return nominal_turn, False, False
    if domain.is_empty:
        return nominal_turn, False, True
    turns = np.unique(np.r_[nominal_turn, np.linspace(-45.0, 45.0, 19)])
    paths = predicted_paths(position, heading, heading + turns, speed_mps,
                            seconds=HORIZON_SECONDS)
    if goal_position is not None:
        paths = stop_at_goal(paths, goal_position, goal_radius)
    trajectories = linestrings(paths)
    feasible = np.asarray(covers(domain, trajectories), dtype=bool)
    goal_error = np.abs((heading + turns - goal_bearing + 180) % 360 - 180)
    if feasible.any():
        choices = np.flatnonzero(feasible)
        # Choose the smallest correction among feasible commands.
        order = np.lexsort((goal_error[choices], np.abs(turns[choices] - nominal_turn)))
        selected = choices[order[0]]
    else:
        # When every candidate violates the geometric margin, favour recovery
        # rather than treating an impossible prediction as a safe command.
        violation = length(difference(trajectories, domain))
        recovery = distance(points(paths[:, -1]), domain)
        cost = violation + 4.0 * recovery
        selected = np.lexsort((goal_error, np.abs(turns - nominal_turn), cost))[0]
    turn = float(turns[selected])
    return turn, turn != nominal_turn, not bool(feasible.any())


class StaticActionProjection:
    """Keep the original observation/reward and filter only direct turn commands."""

    def __init__(self, *args, **kwargs):
        self._projection_domain = None
        self.last_projection = {}
        self.projection_statistics = {"actions": 0, "interventions": 0, "no_feasible_candidate": 0}
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self._projection_domain = None
        self.last_projection = {}
        self.projection_statistics = {"actions": 0, "interventions": 0, "no_feasible_candidate": 0}
        return super().reset(*args, **kwargs)

    def _projection_xy(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=float)
        north = (coordinates[..., 0] - self.scenario.center[0]) * 60 * 1.852
        east = ((coordinates[..., 1] - self.scenario.center[1]) * 60 * 1.852
                * np.cos(np.deg2rad(self.scenario.center[0])))
        return np.stack((east, north), axis=-1)

    def _get_action(self, ac_id, action):
        import bluesky as bs
        from core.tools import kwikqdrdist
        if self._projection_domain is None:
            sector = Polygon(self._projection_xy(self.scenario.sector)).buffer(-CLEARANCE_KM)
            obstacles = unary_union([Polygon(self._projection_xy(obstacle.vertices)).buffer(CLEARANCE_KM)
                                     for obstacle in self.scenario.obstacles])
            self._projection_domain = sector.difference(obstacles)
        idx = bs.traf.id2idx(ac_id)
        goal = self._goal[ac_id] if hasattr(self, "_goal") else (self.goal_lat, self.goal_lon)
        bearing, _ = kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], *goal)
        turn, changed, infeasible = project_turn(
            self._projection_domain, self._projection_xy((bs.traf.lat[idx], bs.traf.lon[idx])),
            float(bs.traf.hdg[idx]), float(action[0]) * 45.0, float(bs.traf.tas[idx]), float(bearing),
            goal_position=self._projection_xy(goal), goal_radius=self.distance_margin)
        filtered = np.asarray(action).copy()
        if changed:
            filtered[0] = turn / 45.0
        self.last_projection[ac_id] = {"commanded_heading_turn_deg": turn,
                                       "static_projection_intervened": changed,
                                       "static_projection_infeasible": infeasible}
        self.projection_statistics["actions"] += 1
        self.projection_statistics["interventions"] += int(changed)
        self.projection_statistics["no_feasible_candidate"] += int(infeasible)
        return super()._get_action(ac_id, filtered)
