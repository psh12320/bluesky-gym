import numpy as np
import pytest

from scripts.analyze_rl_cohort import aggregate_effects, paired_effect, screen


def test_world_pairing_preserves_between_world_variation():
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    effect = paired_effect([0, 10], [10, 10], indices)
    assert effect["mean_difference"] == -5
    assert effect["paired_world_bootstrap_95_interval"] == pytest.approx([-9.625, -0.375])


@pytest.mark.parametrize("current,reference", [
    ([0], [0, 1]), ([], []), ([np.nan, 0], [0, 0]), ([0, 0], [0, np.inf])
])
def test_invalid_physical_records_cannot_produce_an_effect(current, reference):
    with pytest.raises(ValueError):
        paired_effect(current, reference, np.array([[0]]))


def rows():
    return [{"seed": seed, "learning_effects": {
        "clean_completion": {"mean_difference": effect}}}
        for seed, effect in zip([50100, 50200, 50300], [.1, -.2, .4])]


def test_seed_aggregation_retains_the_failure_and_reports_sample_variability():
    result = aggregate_effects(rows(), [50100, 50200, 50300])["clean_completion"]
    assert result["mean_learning_change"] == pytest.approx(.1)
    assert result["learning_change_sample_standard_deviation"] == pytest.approx(.3)
    assert result["per_seed_learning_changes"]["50200"] == -.2


@pytest.mark.parametrize("selection", [[0, 2], [0, 0, 2], [2, 1, 0]])
def test_missing_duplicate_or_reordered_seeds_are_rejected(selection):
    with pytest.raises(ValueError):
        aggregate_effects([rows()[i] for i in selection], [50100, 50200, 50300])


def test_screen_uses_learning_gain_and_rejects_safety_regression():
    gate = dict(arrival_at_least=.95, arrival_change_at_least=-.02,
                clean_completion_change_at_least=.05,
                safety_time_changes_at_most=0, metrics=["intrusion_time"])
    initial = dict(waypoint_reached=1., clean_completion=.90, intrusion_time=2.)
    final = dict(waypoint_reached=1., clean_completion=.95, intrusion_time=2.)
    assert all(screen(initial, final, gate).values())
    final["intrusion_time"] = 2.01
    assert not all(screen(initial, final, gate).values())
    final = dict(initial)
    assert not screen(initial, final, gate)["clean_change"]
