import numpy as np
from shapely import covers, linestrings


def predicted_paths(position, heading, targets, speed_mps, seconds=45.0, dt=5.0, turn_rate=1.5):
    """Approximate bounded-turn paths for checking static geometry.

    This is an action-selection heuristic, not the BlueSky dynamics
    model and not a formal guarantee. Official scoring always uses BlueSky.
    """
    targets = np.asarray(targets, dtype=float)
    headings = np.full(len(targets), heading, dtype=float)
    position = np.broadcast_to(np.asarray(position, dtype=float), (len(targets), 2)).copy()
    points = [position.copy()]
    for _ in range(int(seconds / dt)):
        difference = (targets - headings + 180) % 360 - 180
        turn = np.clip(difference, -turn_rate * dt, turn_rate * dt)
        midpoint = np.deg2rad(headings + 0.5 * turn)
        position += speed_mps * dt / 1000 * np.column_stack((np.sin(midpoint), np.cos(midpoint)))
        headings += turn
        points.append(position.copy())
    return np.stack(points, axis=1)


def filter_heading(domain, position, heading, route_bearing, nominal_offset, speed_mps, turn_limit=45.0):
    """Prefer the learned offset among candidates whose projected paths are free."""
    offsets = np.unique(np.r_[np.clip(nominal_offset, -45.0, 45.0), np.linspace(-45.0, 45.0, 19), 0.0])
    targets = route_bearing + offsets
    turns = np.clip((targets - heading + 180) % 360 - 180, -turn_limit, turn_limit)
    command_targets = heading + turns
    paths = predicted_paths(position, heading, command_targets, speed_mps)
    feasible = np.asarray(covers(domain, linestrings(paths)), dtype=bool)
    if not feasible.any():
        # Recover toward the known navigation reference when the approximation
        # cannot find a feasible action. This can still incur real violations.
        return 0.0
    cost = np.abs(offsets - nominal_offset)
    cost[~feasible] = np.inf
    return float(offsets[np.argmin(cost)])
