import numpy as np
import pytest
from shapely.geometry import box

from atc.traffic_projection import (
    HORIZON_SECONDS, SAMPLE_SECONDS, capture_times, choose_command,
    configuration, predict_commands, traffic_risk,
)
from atc.submission import _validate_configuration


def speed_target(command):
    return 200 + np.asarray(command) * 3.4


def test_joint_filter_preserves_clear_nominal_action_exactly():
    nominal = np.array([.123456789, -.234567891])
    chosen, _, _, info = choose_command(box(-100, -100, 100, 100), [0, 0], 90, 200,
                                        nominal, speed_target)
    assert np.array_equal(chosen, nominal)
    assert not info['nominal_static_violation']
    assert not info['nominal_traffic_conflict']
    assert not info['no_jointly_feasible_candidate']


def test_segment_crossing_is_detected_between_safe_sample_endpoints():
    paths = np.array([[[-10., 0], [10., 0]]])
    others = np.zeros((1, 2, 2))
    safe, risk, minimum = traffic_risk(paths, [5], others, [5], 5)
    assert not safe[0] and risk[0] > 0
    assert minimum[0] == pytest.approx(0)
    # The other aircraft disappears before the paths get within 5 km.
    safe, risk, minimum = traffic_risk(paths, [5], others, [1], 5)
    assert safe[0] and risk[0] == 0
    assert minimum[0] == pytest.approx(6)


def test_goal_capture_uses_strict_entry_and_handles_tangency():
    paths = np.array([[[0., 0], [10., 0]], [[0., 5], [10., 5]], [[13., 0], [20., 0]]])
    times = capture_times(paths, [8, 0], 5)
    assert times[0] == pytest.approx(1.5)
    assert times[1] == HORIZON_SECONDS + SAMPLE_SECONDS
    assert times[2] == HORIZON_SECONDS + SAMPLE_SECONDS


def test_head_on_conflict_uses_a_feasible_turn_and_respects_a_wall():
    other = predict_commands([40, 0], 270, 200, [0], [200])
    chosen, _, _, info = choose_command(box(-100, -.1, 100, 100), [0, 0], 90, 200,
        [0, 0], speed_target, other, [100])
    assert info['nominal_traffic_conflict']
    assert not info['no_jointly_feasible_candidate']
    assert -1 <= chosen[0] < 0  # turn north; the southern wall rules out a right turn
    assert abs(chosen[1]) <= 1
    assert info['predicted_minimum_separation_km'] >= 10.26


def test_preexisting_conflict_chooses_escape_and_reports_infeasibility():
    other = predict_commands([2, 0], 90, 200, [0], [200])
    nominal = predict_commands([0, 0], 90, 200, [0], [200])
    _, old_risk, _ = traffic_risk(nominal, [100], other, [100], 10.26)
    chosen, _, _, info = choose_command(box(-100, -100, 100, 100), [0, 0], 90, 200,
        [0, 0], speed_target, other, [100])
    assert info['no_jointly_feasible_candidate']
    assert not info['no_static_feasible_candidate']
    assert abs(chosen[0]) > 0
    assert info['predicted_risk_seconds'] < old_risk[0]


def test_speed_change_provides_recovery_in_a_narrow_corridor():
    other = predict_commands([10, 0], 90, 200, [0], [200])
    chosen, _, _, info = choose_command(box(-100, -.1, 100, .1), [0, 0], 90, 200,
        [0, 0], speed_target, other, [100])
    assert chosen[0] == 0
    assert chosen[1] < 0
    assert not info['no_static_feasible_candidate']


def test_empty_domain_is_finite_and_explicitly_infeasible():
    chosen, _, _, info = choose_command(box(0, 0, 1, 1).buffer(-2), [0, 0], 90, 200,
                                       [.2, .1], speed_target)
    assert np.isfinite(chosen).all() and np.all(np.abs(chosen) <= 1)
    assert info['no_static_feasible_candidate'] and info['no_jointly_feasible_candidate']


def test_submission_requires_matching_joint_prediction_configuration():
    from atc.projection import PROJECTION_REVISION, HORIZON_SECONDS, CLEARANCE_KM
    config = dict(env='ma', algorithm='sac', recipe='public_weights', guard_static=True,
        guard_traffic=True, static_projection_revision=PROJECTION_REVISION,
        static_projection_horizon_seconds=HORIZON_SECONDS, static_projection_clearance_km=CLEARANCE_KM,
        **configuration())
    _validate_configuration('ma', config)
    config['traffic_projection_margin_km'] += 1
    with pytest.raises(ValueError, match='traffic projection settings'):
        _validate_configuration('ma', config)
