import numpy as np
import pytest

from atc.conflicts import conflict_features


def test_head_on_prediction_matches_analytic_entry_and_closest_approach():
    values = conflict_features([[30000, 0]], [[-300, 0]], [True])
    assert values["traffic_present"][0] == 1
    assert values["traffic_tcpa"][0] == pytest.approx(100 / 180)
    assert values["traffic_dcpa"][0] == pytest.approx(0)
    assert values["traffic_entry_time"][0] == pytest.approx((30000 - 9260) / 300 / 180)
    assert values["traffic_predicted_conflict"][0] == 1


def test_diverging_stationary_and_absent_traffic_are_distinct_and_finite():
    values = conflict_features([[30000, 0], [5000, 0], [0, 0], [0, 0]],
                               [[300, 0], [0, 0], [0, 0], [0, 0]],
                               [True, True, True, False])
    assert np.array_equal(values["traffic_predicted_conflict"], [0, 1, 1, 0])
    assert np.array_equal(values["traffic_entry_time"], [1, 0, 0, 0])
    assert np.array_equal(values["traffic_present"], [1, 1, 1, 0])
    assert values["traffic_dcpa"][0] == pytest.approx(30000 / 9260)
    for value in values.values():
        assert np.all(np.isfinite(value))
        assert value[-1] == 0


def test_predictions_are_rotation_invariant_and_use_only_the_finite_horizon():
    position = np.array([[30000, 5000], [100000, 0], [30000, 9260]], dtype=float)
    velocity = np.array([[-300, 0], [-300, 0], [-300, 0]], dtype=float)
    present = [True, True, True]
    original = conflict_features(position, velocity, present)
    angle = np.deg2rad(63)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = conflict_features(position @ rotation.T, velocity @ rotation.T, present)
    for key in original:
        assert np.allclose(original[key], rotated[key])
    assert original["traffic_dcpa"][0] == pytest.approx(5000 / 9260)
    assert original["traffic_tcpa"][1] == 1
    assert original["traffic_predicted_conflict"][1] == 0
    # A tangent touch at exactly the separation radius is not an intrusion.
    assert original["traffic_predicted_conflict"][2] == 0
