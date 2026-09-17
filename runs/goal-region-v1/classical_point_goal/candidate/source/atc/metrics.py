import numpy as np

METRICS = (
    "waypoint_reached", "flight_time", "intrusion_events", "intrusion_time",
    "restricted_area_events", "time_in_restricted_area",
    "sector_exit_events", "time_outside_sector", "total_reward",
)
SAFETY = (
    "intrusion_events", "intrusion_time", "restricted_area_events",
    "time_in_restricted_area", "sector_exit_events", "time_outside_sector",
)


def summarize(records, expected_episodes, agents_per_episode):
    if not records:
        raise ValueError("No completed aircraft records")
    groups = {}
    for record in records:
        key = (record["episode"], record["agent"])
        group = groups.setdefault(record["episode"], {})
        if record["agent"] in group:
            raise ValueError(f"Duplicate aircraft record: {key}")
        group[record["agent"]] = record
        values = np.array([record[k] for k in METRICS], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite metric in {key}")
        if record["waypoint_reached"] not in (0, 1):
            raise ValueError(f"Invalid completion flag in {key}")
    if set(groups) != set(range(expected_episodes)):
        raise ValueError("Missing or unexpected scenario records")
    if any(len(g) != agents_per_episode for g in groups.values()):
        raise ValueError("Missing or unexpected aircraft records")
    clean = lambda r: bool(r["waypoint_reached"] and all(r[k] == 0 for k in SAFETY))
    return {
        "episodes": expected_episodes,
        "agent_episodes": len(records),
        "metrics": {
            k: {"mean": float(np.mean([r[k] for r in records])),
                "std": float(np.std([r[k] for r in records]))}
            for k in METRICS
        },
        "clean_completion_rate": float(np.mean([clean(r) for r in records])),
        "all_aircraft_clean_completion_rate": float(np.mean([
            all(clean(r) for r in group.values()) for group in groups.values()
        ])),
    }
