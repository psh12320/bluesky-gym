import numpy as np
from shapely.geometry import LineString, Polygon

from atc.routes import VisibilityPlanner


def test_visibility_route_stays_inside_sector_and_avoids_buffered_obstacle():
    sector = [(-20, -20), (20, -20), (20, 20), (-20, 20)]
    obstacle = [(-2, -5), (2, -5), (2, 5), (-2, 5)]
    planner = VisibilityPlanner(sector, [obstacle], clearance=2.0)
    path = planner.path((-15, 0), (15, 0))
    assert path is not None and len(path) > 2
    line = LineString(path)
    assert planner.domain.buffer(1e-7).covers(line)
    assert line.distance(Polygon(obstacle)) >= 2.0 - 1e-7
    assert np.allclose(planner.target(path[0], path), path[1])
    assert np.allclose(planner.target(path[-2], path), path[-1])


def test_clear_path_is_direct_and_disconnected_path_is_rejected():
    sector = [(-20, -20), (20, -20), (20, 20), (-20, 20)]
    planner = VisibilityPlanner(sector, [], clearance=2.0)
    assert len(planner.path((-15, 0), (15, 0))) == 2
    wall = [(-2, -25), (2, -25), (2, 25), (-2, 25)]
    planner = VisibilityPlanner(sector, [wall], clearance=2.0)
    assert planner.path((-15, 0), (15, 0)) is None


def test_sector_inset_keeps_planned_path_away_from_sector_edge():
    sector = [(-20, -20), (20, -20), (20, 20), (-20, 20)]
    obstacle = [(-2, -12), (2, -12), (2, 12), (-2, 12)]
    planner = VisibilityPlanner(sector, [obstacle], clearance=2.0, sector_clearance=3.0)
    path = planner.path((-15, 0), (15, 0))
    assert path is not None
    assert LineString(path).distance(Polygon(sector).boundary) >= 3.0 - 1e-7


def test_evasive_deviation_does_not_reacquire_a_passed_waypoint():
    sector = [(-20, -20), (20, -20), (20, 20), (-20, 20)]
    obstacle = [(-2, -5), (2, -5), (2, 5), (-2, 5)]
    planner = VisibilityPlanner(sector, [obstacle], clearance=2.0)
    path = planner.path((-15, 0), (15, 0))
    first = planner.target_index(path[0], path)
    advanced = planner.target_index(path[first], path, first)
    assert advanced > first
    # The old stateless lookup selected the earlier vertex after this deviation.
    assert planner.target_index(path[0], path) == first
    assert planner.target_index(path[0], path, advanced) == advanced
