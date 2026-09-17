"""Audit all four support cells before estimating PPO learning and support effects."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path(__file__).resolve().parent
GROUPS = ("g0-f0", "g1-f0", "g0-f1", "g1-f1")
STAGES = ("initial", "final", "learning")
CONTRASTS = {
    "guidance_without_filter": {"g1-f0": 1, "g0-f0": -1},
    "guidance_with_filter": {"g1-f1": 1, "g0-f1": -1},
    "filter_without_guidance": {"g0-f1": 1, "g0-f0": -1},
    "filter_with_guidance": {"g1-f1": 1, "g1-f0": -1},
    "guidance_average": {"g1-f0": .5, "g0-f0": -.5, "g1-f1": .5, "g0-f1": -.5},
    "filter_average": {"g0-f1": .5, "g0-f0": -.5, "g1-f1": .5, "g1-f0": -.5},
    "interaction": {"g1-f1": 1, "g1-f0": -1, "g0-f1": -1, "g0-f0": 1},
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(directory):
    manifest = read(directory / "cluster-manifest.json")["files"]
    return {
        key: value for key, value in manifest.items()
        if key.endswith(".py") and key.split("/")[0] in
        {"atc", "atc_rl", "core", "bluesky_gym", "bluesky_zoo"}
    }


def world_effect(values, indices):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite paired whole-world differences")
    return {
        "mean": float(values.mean()),
        "paired_world_bootstrap_95_interval":
            np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist(),
    }


def across_seeds(values, seeds, indices):
    if set(values) != set(seeds) or len(seeds) < 2:
        raise ValueError("Every registered training seed is required")
    matrix = np.stack([np.asarray(values[seed], dtype=float) for seed in seeds])
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Expected finite seed-by-world effects")
    means = matrix.mean(axis=1)
    return {
        "mean": float(means.mean()),
        "sample_standard_deviation_across_training_seeds": float(means.std(ddof=1)),
        "per_seed": {str(seed): world_effect(values[seed], indices) for seed in seeds},
        "paired_world_interval_for_fixed_seed_mean":
            world_effect(matrix.mean(axis=0), indices)["paired_world_bootstrap_95_interval"],
    }


def contrast(cells, weights):
    if set(cells) != set(GROUPS):
        raise ValueError("All four support cells are required for factorial analysis")
    arrays = {group: np.asarray(values, dtype=float) for group, values in cells.items()}
    shape = arrays[GROUPS[0]].shape
    if len(shape) != 1 or not shape[0] or any(a.shape != shape or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Support cells need matching finite world arrays")
    return sum(weight * arrays[group] for group, weight in weights.items())


def audit_rows(plan, existing_only):
    sys.path.insert(0, plan["training_source"])
    from atc_rl.audit import audit
    from atc_rl.algorithm_compare import initial_for
    from atc_rl.checkpoint_identity import verified_checkpoint, policy_fingerprint
    from atc_rl.cluster import verify
    from atc_rl.compare import load_evaluation

    training_source = Path(plan["training_source"])
    evaluation_source = Path(plan["evaluation_source"])
    verify(training_source)
    verify(evaluation_source)
    if digest(ROOT / "runs/onpolicy-learning-source-v2.zip") != plan["training_bundle_sha256"]:
        raise ValueError("Frozen training archive changed")
    prior_record = plan["prior_three_cell_results"]
    if digest(prior_record["path"]) != prior_record["sha256"]:
        raise ValueError("Prior three-cell results changed")
    prior_path = Path(prior_record["path"])
    if digest(prior_path.parent / "protocol.json") != prior_record["protocol_sha256"]:
        raise ValueError("Prior three-cell protocol changed")
    prior = read(prior_path)

    all_keys = [(row["seed"], row["group"]) for row in plan["rows"]]
    expected_keys = {(seed, group) for seed in plan["seeds"] for group in GROUPS}
    if len(all_keys) != 12 or len(set(all_keys)) != 12 or set(all_keys) != expected_keys:
        raise ValueError("The plan must contain all twelve unique rows")
    streams = [world for seed in plan["seeds"] for world in plan["world_seeds"][str(seed)]]
    if len(streams) != len(set(streams)):
        raise ValueError("Training generator streams overlap")
    expected_training = {
        **source_files(training_source),
        "pyproject.toml": digest(training_source / "pyproject.toml"),
    }
    expected_evaluation = source_files(evaluation_source)
    fixed = lambda files: {k: v for k, v in files.items() if k.split("/")[0] != "atc_rl"}
    classical = load_evaluation(Path(plan["classical_evaluation"]))
    if fixed(classical[1]["source_sha256"]) != fixed(expected_evaluation):
        raise ValueError("Classical and learned runs changed the fixed environment/controller")
    worlds = plan["evaluation_worlds"]
    if classical[1]["seed"] != plan["evaluation_seed"] or classical[1]["episodes"] != worlds:
        raise ValueError("Wrong classical evaluation stream")
    if classical[0]["agent_episodes"] != 10 * worlds or set(classical[2]) != set(range(worlds)):
        raise ValueError("Incomplete classical evaluation")
    canonical_config = canonical_environment = None
    tensor_ids = {}
    rows = []
    arrays = {}
    for row in plan["rows"]:
        if row["new_training"] != (row["group"] == "g0-f0"):
            raise ValueError("Only the missing support cell may be new")
        if row["group"] != f"g{int(row['guidance'])}-f{int(row['filter'])}":
            raise ValueError("Support label differs from the recorded flags")
        if existing_only and row["new_training"]:
            continue
        training = Path(row["training"])
        checked = audit(training)
        budget = plan["config"]["live_steps"]
        slots = plan["config"]["workers"] * 10 * plan["config"]["rollout_steps"]
        if checked["status"] != "complete" or not budget <= checked["live_transitions"] < budget + slots:
            raise ValueError("Incomplete or mismatched live training budget")
        config = read(training / "config.json")
        expected = {
            **plan["config"], "seed": row["seed"],
            "world_seeds": plan["world_seeds"][str(row["seed"])],
            "guidance": row["guidance"], "filter": row["filter"],
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Recipe mismatch for {row['seed']} {row['group']}")
        comparable = {k: v for k, v in config.items()
                      if k not in {"seed", "world_seeds", "guidance", "filter", "run_dir"}}
        if canonical_config is None:
            canonical_config = comparable
        if comparable != canonical_config:
            raise ValueError("A setting besides support, seed or output path changed")
        provenance = read(training / "provenance.json")
        if provenance["source_sha256"] != expected_training:
            raise ValueError("Training source is not the registered frozen source")
        environment = {k: provenance[k] for k in ("python", "packages")}
        if canonical_environment is None:
            canonical_environment = environment
        if environment != canonical_environment:
            raise ValueError("Training dependency versions differ")
        if row["new_training"]:
            if read(training.parent / "launch.json")["protocol_sha256"] != digest(PARENT / "protocol.json"):
                raise ValueError("The protocol changed after launch")
        runtime = read(training / "runtime.json")
        for key, value in {
            "population": 10, "decision_interval_seconds": 5,
            "episode_time_limit": 3000, "training_progress_scale": 0.,
            "action_reference": "goal_offset",
        }.items():
            if runtime.get(key) != value:
                raise ValueError("Training runtime changed " + key)
        loaded = {}
        for stage in ("initial", "final"):
            evaluation = load_evaluation(Path(row[stage + "_evaluation"]))
            summary, protocol, scenarios, values = evaluation
            if protocol["source_sha256"] != expected_evaluation:
                raise ValueError("Evaluation source differs from the frozen evaluator")
            if protocol["seed"] != plan["evaluation_seed"] or protocol["episodes"] != worlds:
                raise ValueError("Evaluation scenarios changed")
            if scenarios != classical[2] or set(scenarios) != set(range(worlds)) or summary["agent_episodes"] != 10 * worlds:
                raise ValueError("Incomplete or unpaired worlds")
            for key in ("algorithm", "guidance", "filter", "action_reference", "initial_action_std", "neutral_action_mean"):
                if protocol[key] != config[key]:
                    raise ValueError("Evaluation changed " + key)
            if protocol["evaluation_reward_scale"] != 1 or protocol["evaluation_progress_scale"] != 0:
                raise ValueError("Evaluation does not use native rewards")
            if protocol["training_reward_scale"] != config["reward_scale"] or protocol["training_progress_scale"] != config["progress_scale"]:
                raise ValueError("Evaluation records the wrong training reward")
            for key in ("population", "decision_interval_seconds", "episode_time_limit",
                        "action_reference", "geo_backend", "performance_module"):
                if summary["runtime"][key] != runtime[key]:
                    raise ValueError("Evaluation runtime changed " + key)
            if any(v.shape != (worlds,) or not np.isfinite(v).all() for v in values.values()):
                raise ValueError("Invalid world-level metric array")
            loaded[stage] = evaluation
        checkpoint = verified_checkpoint(training, loaded["final"][1]["checkpoint"])
        if Path(loaded["final"][1]["model_path"]).resolve() != checkpoint:
            raise ValueError("Final checkpoint path differs from its record")
        if loaded["final"][1]["checkpoint"]["sha256"] != checked["model_sha256"]:
            raise ValueError("Evaluated final checkpoint differs from the audited model")
        own_initial = initial_for(checkpoint, loaded["initial"][1])
        identity = policy_fingerprint(own_initial)
        seed_id = tensor_ids.setdefault(row["seed"], identity)
        if identity != seed_id:
            raise ValueError("Full initial policy tensors differ within a training seed")
        stage_arrays = {stage: data[3] for stage, data in loaded.items()}
        stage_arrays["learning"] = {
            metric: stage_arrays["final"][metric].astype(float) - stage_arrays["initial"][metric].astype(float)
            for metric in stage_arrays["initial"]
        }
        arrays[(row["seed"], row["group"])] = stage_arrays
        result = {
            "seed": row["seed"], "group": row["group"], "new_training": row["new_training"],
            "training": str(training), "audit": checked, "full_initial_policy_fingerprint": identity,
            "means": {stage: {metric: float(v.mean()) for metric, v in values.items()}
                      for stage, values in stage_arrays.items()},
            "csv_sha256": {stage: data[0]["csv_sha256"] for stage, data in loaded.items()},
            "training_summary": read(training / "training_summary.json"),
        }
        if not row["new_training"]:
            old = next(r for r in prior["rows"] if r["seed"] == row["seed"] and r["group"] == row["group"])
            for stage in ("initial", "final"):
                if result["means"][stage] != old[stage] or result["csv_sha256"][stage] != old[stage + "_csv_sha256"]:
                    raise ValueError("Recomputed prior metrics or CSV identity changed")
        rows.append(result)
        print(json.dumps({"verified_seed": row["seed"], "group": row["group"]}), flush=True)
    expected_count = 9 if existing_only else 12
    if len(rows) != expected_count:
        raise ValueError("Missing registered rows")
    return rows, arrays, classical, canonical_environment, canonical_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-existing", action="store_true",
                        help="Verify only the nine preserved runs; do not compute incomplete factorial effects.")
    args = parser.parse_args()
    plan = read(PARENT / "protocol.json")
    destination = PARENT / ("existing-nine-audit.json" if args.audit_existing else "four-cell-results.json")
    if destination.exists():
        raise FileExistsError(destination)
    if not args.audit_existing and not (PARENT / "g0-f0/complete.json").is_file():
        raise ValueError("Wait for all three new rows; partial factorial conclusions are disabled")
    rows, arrays, classical, environment, config = audit_rows(plan, args.audit_existing)
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest(PARENT / "protocol.json"),
        "analyzer_sha256": digest(Path(__file__)),
        "verified_rows": len(rows), "all_twelve_rows_included": not args.audit_existing,
        "full_initial_tensors_identical_within_each_verified_seed": True,
        "existing_nine_results_recomputed_and_unchanged": True,
        "training_environment": environment, "common_configuration": config,
        "classical_means": {k: float(v.mean()) for k, v in classical[3].items()},
        "rows": rows, "unseen_scenarios_used": False,
        "limitations": plan["limitations"] + [
            "World bootstrap intervals are conditional on the observed training seeds; they are not training-population confidence intervals.",
            "Across-seed sample standard deviations summarize only three training seeds.",
            "All physical metrics and contrasts are exploratory; no multiple-testing-adjusted significance or general algorithm ranking is claimed.",
        ],
    }
    if not args.audit_existing:
        seeds = plan["seeds"]
        indices = np.random.default_rng(701).integers(
            0, plan["evaluation_worlds"], size=(10000, plan["evaluation_worlds"]))
        metrics = list(arrays[(seeds[0], GROUPS[0])]["initial"])
        result["group_results"] = {
            stage: {
                group: {metric: across_seeds(
                    {seed: arrays[(seed, group)][stage][metric] for seed in seeds}, seeds, indices)
                    for metric in metrics}
                for group in GROUPS}
            for stage in STAGES}
        result["factorial_effects"] = {
            stage: {
                name: {metric: across_seeds({
                    seed: contrast({group: arrays[(seed, group)][stage][metric] for group in GROUPS}, weights)
                    for seed in seeds}, seeds, indices) for metric in metrics}
                for name, weights in CONTRASTS.items()}
            for stage in STAGES}
        result["contrast_coefficients"] = CONTRASTS
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"result": str(destination), "verified_rows": len(rows),
                      "all_twelve_rows_included": result["all_twelve_rows_included"]}))


if __name__ == "__main__":
    main()
