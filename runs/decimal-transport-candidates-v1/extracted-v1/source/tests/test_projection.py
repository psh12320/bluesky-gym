import numpy as np
import pytest
from shapely import covers, linestrings
from shapely.geometry import box

from atc.projection import HORIZON_SECONDS, project_turn
from atc.static_filter import predicted_paths


def test_safe_nominal_command_is_preserved_exactly():
    nominal = 7.123456789
    turn, changed, infeasible = project_turn(box(-100, -100, 100, 100), [0, 0], 90, nominal, 200, 90)
    assert turn == nominal
    assert not changed and not infeasible


def test_projected_command_respects_turn_limit_and_clears_a_wall():
    domain = box(-100, -100, 15, 100)
    turn, changed, infeasible = project_turn(domain, [0, 0], 90, 0, 200, 0)
    assert changed and not infeasible
    assert -45 <= turn < 0
    trajectory = predicted_paths([0, 0], 90, [90 + turn], 200, seconds=HORIZON_SECONDS)
    assert covers(domain, linestrings(trajectory))[0]


def test_infeasible_recovery_is_explicit_and_finite():
    turn, changed, infeasible = project_turn(box(-10, -10, 10, 10), [30, 0], 90, 0, 200, -90)
    assert infeasible and changed
    assert np.isfinite(turn) and abs(turn) <= 45
    turn, changed, infeasible = project_turn(box(0, 0, 1, 1).buffer(-2), [0, 0], 90, 3, 200, 0)
    assert turn == 3 and not changed and infeasible


def test_goal_tracker_uses_field_order_and_preserves_single_and_batch_shapes():
    from gymnasium import spaces
    from atc.baselines import goal_tracker
    dictionary = spaces.Dict({"airspeed": spaces.Box(-10, 10, (2,)),
                              "cos_drift": spaces.Box(-1, 1, (1,)),
                              "sin_drift": spaces.Box(-1, 1, (1,))})
    predict = goal_tracker(dictionary)
    observations = np.array([[9, 9, 1, 0], [9, 9, 0, 1], [9, 9, 0, -1]], dtype=float)
    assert np.array_equal(predict(observations), [[0, 0], [-1, 0], [1, 0]])
    assert np.array_equal(predict(observations[1]), [-1, 0])


def test_prediction_ends_at_goal_capture_before_a_later_boundary_crossing():
    from atc.projection import stop_at_goal
    domain = box(-100, -100, 10, 100)
    # The aircraft reaches its 5 km goal circle at x=3, before the wall at x=10.
    turn, changed, infeasible = project_turn(domain, [0, 0], 90, 0, 200, 90,
                                             goal_position=[8, 0], goal_radius=5)
    assert turn == 0 and not changed and not infeasible
    paths = np.array([[[0, 0], [2, 0], [4, 0], [12, 0]],
                      [[0, 20], [2, 20], [4, 20], [12, 20]]], dtype=float)
    stopped = stop_at_goal(paths, [8, 0], 5)
    assert np.array_equal(stopped[0], [[0, 0], [2, 0], [3, 0], [3, 0]])
    assert np.array_equal(stopped[1], paths[1])
    assert np.array_equal(paths[0, -1], [12, 0])


def test_goal_tangency_and_departure_from_boundary_do_not_end_a_prediction():
    from atc.projection import stop_at_goal
    paths = np.array([[[-2, 5], [0, 5], [2, 5], [12, 5]],
                      [[5, 0], [6, 0], [7, 0], [12, 0]]], dtype=float)
    assert np.array_equal(stop_at_goal(paths, [0, 0], 5), paths)


def test_goal_tracker_constant_speed_keeps_heading_and_batch_layout():
    from gymnasium import spaces
    from atc.baselines import goal_tracker
    dictionary = spaces.Dict({'cos_drift': spaces.Box(-1., 1., (1,)),
                              'sin_drift': spaces.Box(-1., 1., (1,))})
    observations = np.array([[1., 0.], [0., 1.], [0., -1.]], dtype=np.float32)
    original = goal_tracker(dictionary)(observations)
    fast = goal_tracker(dictionary, speed_action=1.0)(observations)
    np.testing.assert_array_equal(fast[:, 0], original[:, 0])
    np.testing.assert_array_equal(fast[:, 1], np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(goal_tracker(dictionary, speed_action=-.25)(observations[0]), [0., -.25])
    for invalid in [float('nan'), float('inf'), -1.01, 1.01]:
        with pytest.raises(ValueError, match='speed_action'):
            goal_tracker(dictionary, speed_action=invalid)
