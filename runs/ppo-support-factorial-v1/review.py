"""Summarize the complete support study without altering its registered analysis."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
source = ROOT / "four-cell-results.json"
r = json.loads(source.read_text(encoding="utf-8"))
assert r["verified_rows"] == 12 and r["all_twelve_rows_included"]
assert r["existing_nine_results_recomputed_and_unchanged"]
groups = ("g0-f0", "g1-f0", "g0-f1", "g1-f1")
labels = ("Neither", "Guidance", "Filter", "Both")
metrics = (
    ("clean_completion", 100., "Clean completion (percentage points)", "Higher is better"),
    ("waypoint_reached", 100., "Arrival (percentage points)", "Higher is better"),
    ("flight_time", 1., "Flight time (seconds)", "Lower is better"),
    ("intrusion_time", 1., "Conflict time (seconds)", "Lower is better"),
    ("time_in_restricted_area", 1., "Restricted-area time (seconds)", "Lower is better"),
    ("time_outside_sector", 1., "Outside-sector time (seconds)", "Lower is better"),
)
checks = 0
for stage in ("initial", "final", "learning"):
    for group in groups:
        rows = [row for row in r["rows"] if row["group"] == group]
        assert len(rows) == 3
        for metric, values in r["group_results"][stage][group].items():
            per_seed = [row["means"][stage][metric] for row in rows]
            np.testing.assert_allclose(np.mean(per_seed), values["mean"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.std(per_seed, ddof=1), values["sample_standard_deviation_across_training_seeds"], atol=1e-12, rtol=0)
            checks += 2

screen = {}
for group in groups:
    final = r["group_results"]["final"][group]
    gain = r["group_results"]["learning"][group]
    conditions = {
        "arrival_at_least_95_percent": final["waypoint_reached"]["mean"] >= .95,
        "arrival_decline_at_most_2_percentage_points": gain["waypoint_reached"]["mean"] >= -.02,
        "clean_completion_gain_at_least_5_percentage_points": gain["clean_completion"]["mean"] >= .05,
        **{m + "_not_increased": gain[m]["mean"] <= 0 for m in (
            "intrusion_time", "time_in_restricted_area", "time_outside_sector")},
    }
    screen[group] = {"checks": conditions, "passes_all": all(conditions.values()),
                     "clean_completion_change_percentage_points": 100 * gain["clean_completion"]["mean"],
                     "across_seed_sample_sd_percentage_points": 100 * gain["clean_completion"]["sample_standard_deviation_across_training_seeds"]}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
colors = ("#286f95", "#4b8e3a", "#a85628")
for ax, (metric, scale, title, direction) in zip(axes.flat, metrics):
    ax.axhline(0, color="#626262", linewidth=1, linestyle="--")
    for i, group in enumerate(groups):
        values = r["group_results"]["learning"][group][metric]
        mean = scale * values["mean"]
        sd = scale * values["sample_standard_deviation_across_training_seeds"]
        ax.errorbar(i, mean, yerr=sd, fmt="s", color="#202020", capsize=5, markersize=5, zorder=3)
        for j, (seed, item) in enumerate(sorted(values["per_seed"].items())):
            ax.scatter(i + (j - 1) * .12, scale * item["mean"], color=colors[j], s=30,
                       label="Seed " + seed if i == 0 else None, zorder=4)
    ax.set_xticks(range(4), labels)
    ax.set_xlim(-.5, 3.5)
    ax.set_title(title + "\n" + direction, fontsize=11)
    ax.grid(axis="y", alpha=.18)
    if metric == "time_outside_sector": ax.set_ylim(-1, 1)
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .105), frameon=False)
fig.suptitle("PPO contribution: trained policy minus its own initial policy", fontsize=16, y=.97)
fig.text(.5, .92, "Four support settings; 3 training seeds each; approximately 100k live transitions; 20 reused development worlds", ha="center", fontsize=10)
fig.text(.5, .077, "Squares and whiskers: mean and sample SD across training seeds (not a confidence interval). Dots: individual seeds.", ha="center", fontsize=9)
fig.text(.5, .048, "Filter combines static-area and aircraft-conflict filtering. Every setting retains goal-offset action mapping.", ha="center", fontsize=9)
fig.text(.5, .019, "Exploratory comparison; no unseen worlds and no claim of general algorithm superiority. All 12 runs are included.", ha="center", fontsize=9)
fig.subplots_adjust(left=.065, right=.98, bottom=.20, top=.86, hspace=.55, wspace=.29)
for suffix in ("png", "pdf"):
    destination = ROOT / ("learning-contribution." + suffix)
    if destination.exists(): raise FileExistsError(destination)
    fig.savefig(destination, dpi=160)
plt.close(fig)
review = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "all_twelve_results_checked": True, "numeric_summary_checks": checks,
    "screen": screen, "any_group_passes_mean_screen": any(row["passes_all"] for row in screen.values()),
    "conclusion": "No support setting demonstrates the required five-percentage-point mean clean-completion gain. High supported scores already occur before PPO training.",
    "physical_metrics": {stage: {group: {metric: {key: value[key] for key in ("mean", "sample_standard_deviation_across_training_seeds")} for metric, value in r["group_results"][stage][group].items()} for group in groups} for stage in ("initial", "final", "learning")},
    "limitations": r["limitations"], "candidate_promoted": False,
    "next_focus": "PPO exploration replication and demonstration-initialized PPO, preserving own-initial controls and the fixed classical benchmark.",
}
with (ROOT / "review.json").open("x", encoding="utf-8") as f: json.dump(review, f, indent=2)
print(json.dumps({"checks": checks, "screen": screen, "candidate_promoted": False}))
