import pytest

from atc.metrics import METRICS, summarize


def record(episode, agent, reached=1, **changes):
    row = dict.fromkeys(METRICS, 0.0)
    row.update(episode=episode, agent=agent, waypoint_reached=reached, **changes)
    return row


def test_aircraft_completion_does_not_imply_clean_team_completion():
    rows = [record(0, "a"), record(0, "b", intrusion_events=1),
            record(1, "a"), record(1, "b", reached=0)]
    result = summarize(rows, 2, 2)
    assert result["metrics"]["waypoint_reached"]["mean"] == 0.75
    assert result["clean_completion_rate"] == 0.5
    assert result["all_aircraft_clean_completion_rate"] == 0.0


def test_missing_and_duplicate_aircraft_are_rejected():
    with pytest.raises(ValueError, match="aircraft records"):
        summarize([record(0, "a")], 1, 2)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([record(0, "a"), record(0, "a")], 1, 2)


def test_nonfinite_results_are_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        summarize([record(0, "a", intrusion_time=float("nan"))], 1, 1)
