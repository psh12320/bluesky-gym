import numpy as np
from shapely.geometry import Polygon, LineString

from atc.static_filter import filter_heading, predicted_paths


def test_projected_turns_respect_heading_wrap_and_speed_units():
    paths = predicted_paths((0, 0), 350, [10], 100, seconds=10, dt=5)
    assert paths.shape == (1, 3, 2)
    assert 0.98 < paths[0, -1, 1] <= 1.0
    assert abs(paths[0, -1, 0]) < 0.1


def test_static_filter_preserves_clear_action_and_rejects_obstacle_turn():
    sector = Polygon([(-20,-20),(20,-20),(20,20),(-20,20)])
    obstacle = Polygon([(1,-1),(10,-1),(10,15),(1,15)])
    domain = sector.difference(obstacle)
    assert filter_heading(domain, (0,0), 0, 0, -20, 150) == -20
    selected = filter_heading(domain, (0,0), 0, 0, 45, 150)
    assert selected < 45
    path = predicted_paths((0,0), 0, [selected], 150)[0]
    assert domain.covers(LineString(path))


def test_no_feasible_prediction_returns_navigation_reference():
    domain=Polygon([(100,100),(110,100),(110,110),(100,110)])
    assert filter_heading(domain, (0,0), 0, 0, 30, 150) == 0
