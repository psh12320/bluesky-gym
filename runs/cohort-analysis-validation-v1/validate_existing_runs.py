from pathlib import Path
import copy
import json
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from scripts.analyze_rl_cohort import analyze, read

seeds = [49920, 49940]
original = read(root / "runs/ppo-guidance-replication-v1/protocol.json")
first = root / f"runs/ppo-guidance-replication-v1/seed-{seeds[0]}/train"
config = read(first / "config.json")
keys = ("algorithm", "workers", "live_steps", "rollout_steps", "batch_size", "epochs",
        "device", "initial_action_std", "neutral_action_mean", "action_reference",
        "guidance", "filter", "progress_scale", "reward_scale",
        "checkpoint_live_steps", "max_wall_seconds")
plan = {
    "seeds": seeds,
    "config": {key: config[key] for key in keys},
    "evaluation": {"worlds": 20, "seed": 20260, "stages": ["initial", "final"]},
    "advance_screen": original["advance_screen"],
}
trials = []
for seed in seeds:
    folder = root / f"runs/ppo-guidance-replication-v1/seed-{seed}"
    trials.append({
        "seed": seed, "training": folder / "train",
        "classical": root / "runs/ppo-direct-pilot-v1/eval-classical-dev20",
        "evaluations": {"initial": folder / "eval-initial-dev20", "final": folder / "eval-final-dev20"},
    })
training_source = root / "runs/onpolicy-learning-source-v2-verify"
evaluation_source = root / "runs/onpolicy-goal-offset-source-v1-verify"
result = analyze(plan, trials, training_source, evaluation_source, root / "runs/onpolicy-cluster-reload-fix-v1-verify")
reference = read(root / "runs/ppo-guidance-replication-v1/three-seed-results.json")
matched = 0
for row in result["per_seed"]:
    expected = next(item for item in reference["per_seed"] if item["seed"] == row["seed"])
    assert row["means"]["initial"] == expected["initial_means"]
    assert row["means"]["final"] == expected["final_means"]
    for metric, effect in row["learning_effects"].items():
        assert abs(effect["mean_difference"] - expected["learning_effects"][metric]["mean_difference"]) < 1e-12
        for left, right in zip(effect["paired_world_bootstrap_95_interval"],
                               expected["learning_effects"][metric]["paired_world_bootstrap_95_percent_interval"]):
            assert abs(left - right) < 1e-12
        matched += 1
    assert row["screen_passed"] == expected["screen_passed"]
with (Path(__file__).parent / "existing-two-seed-validation.json").open("x", encoding="utf-8") as stream:
    json.dump({"purpose": "Analyzer validation against already completed results, not a new cohort selection.",
               "matched_seed_metric_effects_and_intervals": matched, "result": result}, stream, indent=2)

partial = root / "runs/goal-offset-scaled-million-retry-v1/train"
partial_config = read(partial / "config.json")
partial_plan = copy.deepcopy(plan)
partial_plan["seeds"] = [49900, 49920]
partial_plan["config"] = {key: partial_config[key] for key in keys}
partial_trials = [
    {**trials[0], "seed": 49900, "training": partial},
    trials[0],
]
try:
    analyze(partial_plan, partial_trials, training_source, evaluation_source, root / "runs/onpolicy-cluster-reload-fix-v1-verify")
except ValueError as error:
    if "did not complete the registered budget" not in str(error):
        raise
    rejection = str(error)
else:
    raise AssertionError("A real wall-limited run was accepted as completed training")
with (Path(__file__).parent / "validation.json").open("x", encoding="utf-8") as stream:
    json.dump({"two_completed_real_runs_audited": True, "all_seed_metric_effects_matched": matched,
               "real_incomplete_million_step_run_rejected": rejection,
               "cluster_baseline_results_received": False,
               "validation_does_not_constitute_new_training": True}, stream, indent=2)
print(json.dumps({"matched_effects_and_intervals": matched, "partial_budget_rejected": rejection}))
