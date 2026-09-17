"""Choose nearby learned commands under joint traffic and static predictions."""
import numpy as np
from shapely import covers, difference, distance, get_point, length, linestrings
from shapely.geometry import Polygon
from shapely.ops import unary_union

from atc.projection import CLEARANCE_KM, StaticActionProjection, stop_at_goal

REVISION = 1
HORIZON_SECONDS = 90.0
SAMPLE_SECONDS = 5.0
TRAFFIC_MARGIN_KM = 1.0
TURN_RATE_DEG_PER_SECOND = 1.5
ACCELERATION_MPS2 = 1.0


def configuration():
    return dict(traffic_projection_revision=REVISION,
                traffic_projection_horizon_seconds=HORIZON_SECONDS,
                traffic_projection_sample_seconds=SAMPLE_SECONDS,
                traffic_projection_margin_km=TRAFFIC_MARGIN_KM,
                traffic_projection_turn_rate=TURN_RATE_DEG_PER_SECOND,
                traffic_projection_acceleration=ACCELERATION_MPS2)


def predict_commands(position, heading, speed_mps, turns, target_speeds):
    """Approximate heading capture and acceleration; positions are in kilometres."""
    turns, targets = np.asarray(turns, dtype=float), np.asarray(target_speeds, dtype=float)
    headings = np.full(len(turns), heading, dtype=float)
    speeds = np.full(len(turns), speed_mps, dtype=float)
    positions = np.broadcast_to(position, (len(turns), 2)).astype(float).copy()
    paths = [positions.copy()]
    for _ in range(round(HORIZON_SECONDS / SAMPLE_SECONDS)):
        rotation = np.clip((heading + turns - headings + 180) % 360 - 180,
                           -TURN_RATE_DEG_PER_SECOND * SAMPLE_SECONDS,
                           TURN_RATE_DEG_PER_SECOND * SAMPLE_SECONDS)
        acceleration = np.clip(targets - speeds, -ACCELERATION_MPS2 * SAMPLE_SECONDS,
                               ACCELERATION_MPS2 * SAMPLE_SECONDS)
        angle = np.deg2rad(headings + rotation / 2)
        travel = (speeds + acceleration / 2) * SAMPLE_SECONDS / 1000
        positions += travel[:, None] * np.column_stack((np.sin(angle), np.cos(angle)))
        headings += rotation
        speeds += acceleration
        paths.append(positions.copy())
    return np.stack(paths, axis=1)


def capture_times(paths, goal=None, radius=5.0):
    """First strict goal entry, in seconds; uncaptured paths live through the horizon."""
    if goal is None:
        return np.full(len(paths), HORIZON_SECONDS + SAMPLE_SECONDS)
    relative, delta = paths[:, :-1] - np.asarray(goal), np.diff(paths, axis=1)
    a = np.sum(delta * delta, axis=-1)
    b = 2 * np.sum(relative * delta, axis=-1)
    c = np.sum(relative * relative, axis=-1) - radius * radius
    disc = b * b - 4 * a * c
    root = (-b - np.sqrt(np.maximum(disc, 0))) / np.where(a > 1e-12, 2 * a, 1)
    inside = c < 0
    entry = inside | ((a > 1e-12) & (disc > 0) & (root >= 0) & (root < 1))
    time = (np.arange(paths.shape[1] - 1) + np.where(inside, 0, root)) * SAMPLE_SECONDS
    return np.min(np.where(entry, time, HORIZON_SECONDS + SAMPLE_SECONDS), axis=1)


def traffic_risk(paths, lifetimes, other_paths, other_lifetimes, separation_km):
    """Check whole linear segments and remove aircraft after predicted goal capture.

    Return per-candidate feasibility, penetration-weighted exposure and minimum
    separation. Prediction errors and later replanning can still cause real conflicts.
    """
    if not len(other_paths):
        return np.ones(len(paths), bool), np.zeros(len(paths)), np.full(len(paths), np.inf)
    other_paths = np.asarray(other_paths, dtype=float)
    relative = paths[:, None] - other_paths[None]
    start, delta = relative[:, :, :-1], np.diff(relative, axis=2)
    end_time = np.minimum(np.asarray(lifetimes)[:, None], np.asarray(other_lifetimes)[None])
    upper = np.clip((end_time[:, :, None] - np.arange(start.shape[2]) * SAMPLE_SECONDS)
                    / SAMPLE_SECONDS, 0, 1)
    active = upper > 0
    a = np.sum(delta * delta, axis=-1)
    b = np.sum(start * delta, axis=-1)
    c = np.sum(start * start, axis=-1) - separation_km ** 2
    moving = a > 1e-12
    fraction = np.clip(np.divide(-b, a, out=np.zeros_like(b), where=moving), 0, upper)
    closest = np.linalg.norm(start + fraction[..., None] * delta, axis=-1)
    minimum = np.min(np.where(active, closest, np.inf), axis=(1, 2))
    disc = b * b - a * c
    root = np.sqrt(np.maximum(disc, 0))
    entry = np.divide(-b - root, a, out=np.zeros_like(b), where=moving)
    leave = np.divide(-b + root, a, out=np.zeros_like(b), where=moving)
    interval = np.maximum(0, np.minimum(leave, upper) - np.maximum(entry, 0))
    interval = np.where(moving & (disc > 0), interval, np.where(~moving & (c < 0), upper, 0))
    exposure = np.sum(interval * SAMPLE_SECONDS, axis=(1, 2))
    penetration = np.maximum(0, 1 - closest / separation_km)
    risk = exposure + np.sum(upper * SAMPLE_SECONDS * penetration, axis=(1, 2))
    return minimum >= separation_km, risk, minimum


def choose_command(domain, position, heading, speed_mps, nominal_action,
                   target_speed, others=(), other_lifetimes=(), goal=None,
                   goal_radius=5.0, separation_km=10.26):
    """Select a nearby turn/speed command, with explicit infeasibility diagnostics.

    target_speed maps normalized speed commands to predicted true airspeed. Static
    feasibility is prioritized if no candidate satisfies both kinds of constraint.
    """
    nominal = np.clip(np.asarray(nominal_action, dtype=float), -1, 1)
    def assess(actions):
        paths = predict_commands(position, heading, speed_mps, actions[:, 0] * 45,
                                 target_speed(actions[:, 1]))
        life = capture_times(paths, goal, goal_radius)
        static_paths = stop_at_goal(paths, goal, goal_radius) if goal is not None else paths
        trajectories = linestrings(static_paths)
        static = np.asarray(covers(domain, trajectories), dtype=bool)
        traffic, risk, minimum = traffic_risk(paths, life, others, other_lifetimes, separation_km)
        return paths, life, trajectories, static, traffic, risk, minimum
    initial = assess(nominal[None])
    nominal_static, nominal_traffic = bool(initial[3][0]), bool(initial[4][0])
    if nominal_static and nominal_traffic:
        chosen, results, index = nominal, initial, 0
    else:
        turns = np.unique(np.r_[nominal[0], np.linspace(-1, 1, 19)])
        speeds = np.unique(np.r_[nominal[1], -1, 0, 1])
        actions = np.array(np.meshgrid(turns, speeds, indexing='ij')).reshape(2, -1).T
        results = assess(actions)
        paths, life, trajectories, static, traffic, risk, minimum = results
        joint = static & traffic
        motion = np.abs(actions[:, 0] - nominal[0]) + .35 * np.abs(actions[:, 1] - nominal[1])
        if joint.any():
            candidates = np.flatnonzero(joint)
            index = candidates[np.lexsort((-actions[candidates, 0], motion[candidates]))[0]]
        elif static.any():
            candidates = np.flatnonzero(static)
            index = candidates[np.lexsort((-actions[candidates, 0], motion[candidates], risk[candidates]))[0]]
        else:
            if domain.is_empty:
                static_cost = np.zeros(len(actions))
            else:
                static_cost = length(difference(trajectories, domain)) + 4 * distance(get_point(trajectories, -1), domain)
            index = np.lexsort((-actions[:, 0], motion, risk, static_cost))[0]
        chosen = actions[index]
    paths, life, _, static, traffic, risk, minimum = results
    diagnostics = dict(nominal_static_violation=not nominal_static, nominal_traffic_conflict=not nominal_traffic,
                       no_static_feasible_candidate=not bool(static.any()),
                       no_jointly_feasible_candidate=not bool((static & traffic).any()),
                       predicted_minimum_separation_km=float(minimum[index]),
                       predicted_risk_seconds=float(risk[index]))
    return chosen, paths[index], float(life[index]), diagnostics


class TrafficActionProjection(StaticActionProjection):
    """Coordinate predictions in the environment's stable aircraft-action order."""

    def __init__(self, *args, **kwargs):
        self._clear_traffic_plans()
        super().__init__(*args, **kwargs)

    def _clear_traffic_plans(self):
        self._traffic_plan_time = None
        self._traffic_plans = {}
        self.traffic_projection_statistics = dict.fromkeys((
            'actions', 'interventions', 'speed_interventions', 'nominal_static_violation',
            'nominal_traffic_conflict', 'no_static_feasible_candidate', 'no_jointly_feasible_candidate'), 0)

    def reset(self, *args, **kwargs):
        self._clear_traffic_plans()
        return super().reset(*args, **kwargs)

    def _get_action(self, ac_id, action):
        import bluesky as bs
        from core.actions import MpS2Kt
        if self._projection_domain is None:
            sector = Polygon(self._projection_xy(self.scenario.sector)).buffer(-CLEARANCE_KM)
            obstacles = unary_union([Polygon(self._projection_xy(obstacle.vertices)).buffer(CLEARANCE_KM)
                                     for obstacle in self.scenario.obstacles])
            self._projection_domain = sector.difference(obstacles)
        decision_time = float(bs.sim.simt)
        if self._traffic_plan_time != decision_time:
            self._traffic_plans = {}
            self._traffic_plan_time = decision_time
            for index, agent in enumerate(bs.traf.id):
                path = predict_commands(self._projection_xy((bs.traf.lat[index], bs.traf.lon[index])),
                                        float(bs.traf.hdg[index]), float(bs.traf.tas[index]),
                                        [0], [float(bs.traf.tas[index])])
                goal = self._goal.get(agent) if hasattr(self, '_goal') else None
                life = capture_times(path, self._projection_xy(goal) if goal is not None else None,
                                     self.distance_margin)[0]
                self._traffic_plans[agent] = (path[0], life)
        index = bs.traf.id2idx(ac_id)
        position = self._projection_xy((bs.traf.lat[index], bs.traf.lon[index]))
        heading, tas, cas = map(float, (bs.traf.hdg[index], bs.traf.tas[index], bs.traf.cas[index]))
        vmin, vmax = float(bs.traf.perf.vmin[index]), float(bs.traf.perf.vmax[index])
        # OpenAP supplies CAS envelope limits. Use the observed TAS/CAS ratio,
        # keeping this approximation horizontal and independent of altitude input.
        def target_speed(command):
            requested = cas + np.asarray(command) * self.speed_action.d_speed / MpS2Kt
            return np.clip(requested, vmin, vmax) * tas / max(cas, 1e-6)
        others = [plan for agent, plan in self._traffic_plans.items() if agent != ac_id]
        other_paths = np.array([plan[0] for plan in others])
        other_lifetimes = np.array([plan[1] for plan in others])
        goal = self._goal[ac_id] if hasattr(self, '_goal') else (self.goal_lat, self.goal_lon)
        selected, path, life, diagnostics = choose_command(
            self._projection_domain, position, heading, tas, action, target_speed,
            other_paths, other_lifetimes, self._projection_xy(goal), self.distance_margin,
            self.intrusion_distance * 1.852 + TRAFFIC_MARGIN_KM)
        self._traffic_plans[ac_id] = (path, life)
        changed = not np.array_equal(selected, np.asarray(action))
        speed_changed = selected[1] != action[1]
        stats = self.traffic_projection_statistics
        stats['actions'] += 1
        stats['interventions'] += int(changed)
        stats['speed_interventions'] += int(speed_changed)
        for key in diagnostics:
            if key in stats:
                stats[key] += int(diagnostics[key])
        self.last_projection[ac_id] = dict(commanded_heading_turn_deg=float(selected[0] * 45),
            commanded_speed_action=float(selected[1]), traffic_projection_intervened=changed,
            static_projection_intervened=changed and diagnostics['nominal_static_violation'],
            static_projection_infeasible=diagnostics['no_static_feasible_candidate'], **diagnostics)
        # Send only the chosen horizontal commands; the original simulator and
        # scoring loop execute them. Reapplying the static-only filter here would
        # discard the joint feasibility calculation.
        self.heading_action.execute(ac_id, selected[0])
        self.speed_action.execute(ac_id, selected[1])
