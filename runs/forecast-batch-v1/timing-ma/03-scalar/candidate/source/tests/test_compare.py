import pytest

from atc.compare import paired_comparison
from atc.metrics import METRICS


def row(episode, agent, intrusion):
    return dict(episode=episode, agent=agent,
                **{k: 1.0 if k == "waypoint_reached" else intrusion if k == "intrusion_time" else 0.0 for k in METRICS})


def test_bootstrap_preserves_correlated_aircraft_within_each_scenario():
    reference = [row(ep, str(ac), 10.0) for ep in range(2) for ac in range(10)]
    candidate = [row(ep, str(ac), 0.0 if ep == 0 else 10.0) for ep in range(2) for ac in range(10)]
    result = paired_comparison(reference, candidate, 2)
    assert result["intrusion_time"]["delta"] == -5.0
    assert result["intrusion_time"]["ci95"] == [-10.0, 0.0]
    assert result["clean_completion_rate"]["candidate"] == 0.5
    assert result["all_aircraft_clean_completion_rate"]["candidate"] == 0.5


def test_mismatched_scenarios_cannot_be_compared_as_pairs():
    with pytest.raises(ValueError, match="do not match"):
        paired_comparison([row(0, "a", 0)], [row(1, "a", 0)], 2)
