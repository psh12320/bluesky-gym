"""Summarize the completed SAC/PPO and GAE pilots without selecting checkpoints."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding="utf-8-sig"))
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    paths = {
        "algorithm": ROOT / "runs/sac-ppo-matched-pilot-v1/paired-results.json",
        "credit": ROOT / "runs/ppo-credit-pilot-v1/paired-results.json",
    }
    reports = {key: read(path) for key, path in paths.items()}
    if not all(report["both_registered_arms_included"] for report in reports.values()):
        raise ValueError("Both comparisons must be completed and audited first")
    if reports["algorithm"]["seed"] != reports["credit"]["seed"]:
        raise ValueError("The comparisons do not share the registered training seed")
    source = ROOT / "runs/sac-comparison-source-v1-verify"
    sys.path.insert(0, str(source))
    from atc_rl.cluster import verify
    from atc_rl.compare import load_evaluation
    verify(source)
    choices = (
        ("SAC", reports["algorithm"]["arms"]["sac"], "#c46d20", "sac"),
        ("PPO, GAE 0.95", reports["algorithm"]["arms"]["ppo"], "#2368a2", "ppo095"),
        ("PPO, GAE 0.99", reports["credit"]["arms"]["lambda099"], "#88539b", "ppo099"),
    )
    old_control = reports["algorithm"]["arms"]["ppo"]
    reused_control = reports["credit"]["arms"]["lambda095"]
    for key in ("means", "learning_effects", "csv_sha256", "curve"):
        if old_control[key] != reused_control[key]:
            raise ValueError("The reused PPO control is not identical")
    rows = []
    classical = None
    for label, arm, color, short in choices:
        curve = arm["curve"]
        if len(curve["points"]) != 3:
            raise ValueError("Missing initial/intermediate/final curve point")
        if classical is None:
            classical = curve["classical_means"]
        if classical != curve["classical_means"]:
            raise ValueError("Fixed classical reference differs")
        for point, stage in zip(curve["points"], ("initial", "50k", "final")):
            evaluation = load_evaluation(Path(point["evaluation"]))
            if evaluation[0]["csv_sha256"] != arm["csv_sha256"][stage]:
                raise ValueError("Raw evaluation changed after the audit")
            for metric, values in evaluation[3].items():
                mean = float(values.mean())
                if mean != point["metrics"][metric]["mean"] or mean != arm["means"][stage][metric]:
                    raise ValueError("Learning curve or audit summary differs from raw metrics")
                rows.append({"policy": short, "stage": stage, "live_transitions": point["live_transitions"],
                             "metric": metric, "mean": mean,
                             "change_from_initial": point["metrics"][metric]["difference_from_initial"]})
    artifacts = OUT / "artifacts"
    artifacts.mkdir(exist_ok=False)
    with (artifacts / "learning-curves.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    metrics = [
        ("waypoint_reached", "Arrival", 100),
        ("clean_completion", "Clean completion", 100),
        ("flight_time", "Flight time (seconds)", 1),
        ("intrusion_time", "Aircraft-conflict time (seconds)", 1),
        ("time_in_restricted_area", "Restricted-area time (seconds)", 1),
        ("time_outside_sector", "Outside-sector time (seconds)", 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    for ax, (metric, title, scale) in zip(axes.flat, metrics):
        for label, arm, color, short in choices:
            points = arm["curve"]["points"]
            x = np.array([p["live_transitions"] / 1000 for p in points])
            y = np.array([p["metrics"][metric]["mean"] * scale for p in points])
            lo = np.array([p["metrics"][metric]["world_bootstrap_95_interval"][0] * scale for p in points])
            hi = np.array([p["metrics"][metric]["world_bootstrap_95_interval"][1] * scale for p in points])
            ax.plot(x, y, marker="o", ms=4, color=color, label=label)
            ax.fill_between(x, lo, hi, alpha=.09, color=color)
        ax.axhline(classical[metric] * scale, linestyle="--", color="#477347", label="Fixed classical")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Live training transitions (thousands)", fontsize=9)
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
        if scale == 100:
            ax.yaxis.set_major_formatter(PercentFormatter(100))
            ax.set_ylim(0 if metric == "clean_completion" else 65, 102)
        elif metric == "time_outside_sector":
            ax.set_ylim(-.02, .25)
        else:
            ax.set_ylim(bottom=0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .935), ncol=4, frameon=False)
    fig.suptitle("SAC/PPO and credit-assignment pilots: one training seed", fontsize=15, y=.985)
    fig.text(.5, .025,
             "20 reused development worlds; bands resample worlds, not training seeds. "
             "All registered checkpoints shown.\n"
             "SAC and PPO use different training recipes. The PPO control is reused for the GAE comparison.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .88))
    fig.savefig(artifacts / "learning-curves.png", dpi=160)
    fig.savefig(artifacts / "learning-curves.pdf")
    plt.close(fig)
    decision = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()},
        "raw_metric_curve_checks": len(rows),
        "seed": reports["algorithm"]["seed"],
        "unique_training_runs": 3,
        "ppo_control_reused_not_an_independent_replication": True,
        "policies": {short: {
            "label": label, "training": arm["audit"],
            "final": arm["means"]["final"], "initial": arm["means"]["initial"],
            "learning_effects": arm["learning_effects"],
            "screen_checks": arm["screen_checks"], "screen_passed": arm["screen_passed"],
        } for label, arm, color, short in choices},
        "sac_minus_ppo_learning": reports["algorithm"]["sac_minus_ppo_learning"],
        "higher_minus_default_gae_learning": reports["credit"]["higher_minus_default_learning"],
        "candidate_promoted": False,
        "unseen_scenarios_used": False,
        "limitations": [
            "One paired training seed and reused development worlds; no general algorithm ranking.",
            "SAC/PPO network and optimization recipes differ; this is not an isolated algorithm-only comparison.",
            "The GAE comparison changes only lambda, with identical full initialization and its reused PPO control.",
            "World intervals do not estimate training-seed variability.",
            "The fixed classical benchmark has different action support; initial-policy controls isolate learning within each recipe.",
        ],
    }
    with (OUT / "decision.json").open("x", encoding="utf-8") as stream:
        json.dump(decision, stream, indent=2)
    print(json.dumps({"decision": str(OUT / "decision.json"), "raw_metric_curve_checks": len(rows),
                      "screens": {short: arm["screen_passed"] for label, arm, color, short in choices}}))


if __name__ == "__main__":
    main()
