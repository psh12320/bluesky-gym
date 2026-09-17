import argparse
import csv
import json
from pathlib import Path

import numpy as np

from atc.metrics import METRICS, SAFETY, summarize


def load_evaluation(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with path.with_suffix(".csv").open(newline="", encoding="utf-8") as stream:
        records = [dict(episode=int(row["episode"]), agent=row["agent"],
                        **{key: float(row[key]) for key in METRICS})
                   for row in csv.DictReader(stream)]
    summarize(records, metadata["episodes"], 1 if metadata["env"] == "sa" else 10)
    return metadata, records


def paired_comparison(reference, candidate, episodes, resamples=10000, seed=0):
    """Bootstrap complete scenarios, preserving within-scenario dependence."""
    keys = METRICS[:-1] + ("clean_completion_rate", "all_aircraft_clean_completion_rate")
    def group(rows):
        result = {}
        for row in rows:
            identity = (row["episode"], row["agent"])
            if identity in result:
                raise ValueError("Duplicate aircraft record")
            result[identity] = row
        return result
    old, new = group(reference), group(candidate)
    if set(old) != set(new):
        raise ValueError("Evaluation aircraft and scenario IDs do not match")
    if {ep for ep, _ in old} != set(range(episodes)):
        raise ValueError("Missing scenario IDs")
    if episodes < 2 or resamples < 100:
        raise ValueError("At least two scenarios and 100 bootstrap samples are required")
    def scenario_values(rows):
        result = []
        for episode in range(episodes):
            group_rows = [r for (ep, _), r in rows.items() if ep == episode]
            clean = [bool(r["waypoint_reached"] and all(r[k] == 0 for k in SAFETY)) for r in group_rows]
            result.append([np.mean([r[k] for r in group_rows]) for k in METRICS[:-1]]
                          + [np.mean(clean), float(all(clean))])
        return np.asarray(result)
    a, b = scenario_values(old), scenario_values(new)
    delta = b - a
    rng = np.random.default_rng(seed)
    boot = np.empty((resamples, len(keys)))
    for start in range(0, resamples, 250):
        size = min(250, resamples - start)
        indices = rng.integers(0, episodes, size=(size, episodes))
        boot[start:start + size] = delta[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    return {key: {"reference": float(a[:, i].mean()), "candidate": float(b[:, i].mean()),
                  "delta": float(delta[:, i].mean()), "ci95": [float(low[i]), float(high[i])],
                  "better": "higher" if key in ("waypoint_reached", "clean_completion_rate", "all_aircraft_clean_completion_rate") else "lower"}
            for i, key in enumerate(keys)}


def main():
    parser = argparse.ArgumentParser(description="Compare policies on matched development scenarios.")
    parser.add_argument("reference", type=Path, help="Evaluation CSV or output prefix")
    parser.add_argument("candidate", type=Path, help="Evaluation CSV or output prefix")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    old_meta, old = load_evaluation(args.reference)
    new_meta, new = load_evaluation(args.candidate)
    for key in ("env", "seed", "episodes"):
        if old_meta[key] != new_meta[key]:
            parser.error(f"Evaluations differ in {key}; paired comparison is invalid")
    if args.out.exists():
        parser.error("Output already exists")
    result = {"reference": str(args.reference), "candidate": str(args.candidate),
              "env": old_meta["env"], "seed": old_meta["seed"], "episodes": old_meta["episodes"],
              "interval": "95% paired scenario bootstrap, pointwise; does not include training-seed uncertainty",
              "metrics": paired_comparison(old, new, old_meta["episodes"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Metric | Reference | Candidate | Change [95% interval]")
    for key, value in result["metrics"].items():
        low, high = value["ci95"]
        print(f"{key} | {value['reference']:.4f} | {value['candidate']:.4f} | "
              f"{value['delta']:+.4f} [{low:+.4f}, {high:+.4f}]")


if __name__ == "__main__":
    main()
