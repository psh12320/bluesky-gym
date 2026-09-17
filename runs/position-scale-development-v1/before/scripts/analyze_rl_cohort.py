"""Analyze every registered seed from completed training and paired evaluations."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_effect(current, reference, indices):
    current, reference = np.asarray(current, dtype=float), np.asarray(reference, dtype=float)
    if current.shape != reference.shape or current.ndim != 1 or not len(current):
        raise ValueError("Paired world arrays must have equal nonzero lengths")
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite physical metrics")
    difference = current - reference
    return {
        "mean_difference": float(difference.mean()),
        "paired_world_bootstrap_95_interval":
            np.quantile(difference[indices].mean(axis=1), [.025, .975]).tolist(),
    }


def aggregate_effects(rows, seeds):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Register at least two distinct training seeds")
    if [row["seed"] for row in rows] != seeds:
        raise ValueError("Missing, duplicate or reordered registered seed")
    metrics = set(rows[0]["learning_effects"])
    if any(set(row["learning_effects"]) != metrics for row in rows):
        raise ValueError("Different reported metrics across seeds")
    result = {}
    for metric in sorted(metrics):
        values = np.array([
            row["learning_effects"][metric]["mean_difference"] for row in rows
        ], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite seed effects")
        result[metric] = {
            "mean_learning_change": float(values.mean()),
            "learning_change_sample_standard_deviation": float(values.std(ddof=1)),
            "per_seed_learning_changes":
                {str(seed): float(value) for seed, value in zip(seeds, values)},
        }
    return result


def screen(initial, final, gate):
    tolerance = 1e-12
    return {
        "arrival_level": final["waypoint_reached"] >= gate["arrival_at_least"] - tolerance,
        "arrival_change": final["waypoint_reached"] - initial["waypoint_reached"]
            >= gate["arrival_change_at_least"] - tolerance,
        "clean_change": final["clean_completion"] - initial["clean_completion"]
            >= gate["clean_completion_change_at_least"] - tolerance,
        **{metric: final[metric] - initial[metric] <= gate["safety_time_changes_at_most"] + tolerance
           for metric in gate["metrics"]},
    }


def analyze(plan, trials, training_source, evaluation_source, classical_source=None):
    """Inspect original artifacts; relocated paths never rewrite recorded metadata."""
    training_source, evaluation_source = Path(training_source).resolve(), Path(evaluation_source).resolve()
    sys.path.insert(0, str(training_source))
    from atc_rl.cluster import verify
    from atc_rl.audit import audit
    from atc_rl.compare import load_evaluation
    from atc_rl.checkpoint_identity import verified_checkpoint
    from atc_rl.matrix_evaluate import select_checkpoint

    verify(training_source)
    verify(evaluation_source)
    classical_source = Path(classical_source).resolve() if classical_source else evaluation_source
    if classical_source != evaluation_source:
        verify(classical_source)
    seeds = plan["seeds"]
    if [trial["seed"] for trial in trials] != seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Trial list must contain every registered seed exactly once")
    stages = plan["evaluation"]["stages"]
    if stages[0] != "initial" or stages[-1] != "final" or len(stages) != len(set(stages)):
        raise ValueError("Register distinct stages from initial through final")
    worlds = plan["evaluation"]["worlds"]
    indices = np.random.default_rng(701).integers(0, worlds, size=(10000, worlds))
    package_names = {"atc", "atc_rl", "core", "bluesky_gym", "bluesky_zoo"}
    source_files = lambda source: {
        name: value for name, value in read(source / "cluster-manifest.json")["files"].items()
        if name.endswith(".py") and name.split("/")[0] in package_names
    }
    expected_training = {"pyproject.toml": digest(training_source / "pyproject.toml"),
                         **source_files(training_source)}
    expected_evaluation = source_files(evaluation_source)
    expected_classical = source_files(classical_source)
    fixed = lambda files: {k: v for k, v in files.items() if k.split('/')[0] != 'atc_rl'}
    if fixed(expected_classical) != fixed(expected_evaluation):
        raise ValueError('Benchmark and learned evaluation changed controller or simulator source')
    canonical_config = canonical_environment = canonical_scenarios = canonical_classical = None
    used_world_seeds = set()
    rows = []

    for trial in trials:
        seed, training = trial["seed"], Path(trial["training"])
        checked, config = audit(training), read(training / "config.json")
        expected = {**plan["config"], "seed": seed}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Training recipe changed for seed {seed}")
        for key, default in (("exploration", "gaussian"), ("sde_weight_std", None), ("sde_sample_freq", None)):
            if config.get(key, default) != plan["config"].get(key, default):
                raise ValueError("Unregistered exploration setting: " + key)
        maximum_rollout = config["workers"] * 10 * config["rollout_steps"]
        if checked["status"] != "complete" or not (
                config["live_steps"] <= checked["live_transitions"] < config["live_steps"] + maximum_rollout):
            raise ValueError(f"Seed {seed} did not complete the registered budget")
        recipe = {key: value for key, value in config.items()
                  if key not in {"run_dir", "seed", "world_seeds"}}
        if canonical_config is None:
            canonical_config = recipe
        if recipe != canonical_config:
            raise ValueError("Unregistered configuration differences across seeds")
        expected_world_seeds = [seed + 10 * index for index in range(config["workers"])]
        if config["world_seeds"] != expected_world_seeds or used_world_seeds.intersection(expected_world_seeds):
            raise ValueError("Wrong or overlapping simulator training seeds")
        used_world_seeds.update(expected_world_seeds)
        provenance = read(training / "provenance.json")
        if provenance["source_sha256"] != expected_training:
            raise ValueError("Training source differs from its frozen reference")
        environment = {key: provenance[key] for key in ("python", "packages")}
        if canonical_environment is None:
            canonical_environment = environment
        if environment != canonical_environment:
            raise ValueError("Different training dependencies across seeds")

        paths = {"classical": Path(trial["classical"]),
                 **{stage: Path(trial["evaluations"][stage]) for stage in stages}}
        evaluated, recorded_points = {}, []
        for stage, folder in paths.items():
            summary, protocol, scenarios, values = load_evaluation(folder)
            if set(scenarios) != set(range(worlds)) or summary["episodes"] != worlds:
                raise ValueError("Incomplete evaluation worlds")
            if summary["agent_episodes"] != worlds * 10:
                raise ValueError("Incomplete aircraft records")
            expected_source = expected_classical if stage == "classical" else expected_evaluation
            if (protocol["seed"] != plan["evaluation"]["seed"]
                    or protocol["episodes"] != worlds or protocol["source_sha256"] != expected_source):
                raise ValueError("Wrong evaluation stream or source")
            if protocol.get("evaluation_reward_scale", 1) != 1 or protocol.get("evaluation_progress_scale", 0) != 0:
                raise ValueError("Evaluation must retain native rewards")
            if canonical_scenarios is None:
                canonical_scenarios = scenarios
            if scenarios != canonical_scenarios:
                raise ValueError("Evaluation scenarios are not paired")
            if stage == "classical":
                if (protocol["algorithm"] != "classical" or not protocol["guidance"]
                        or not protocol["filter"] or protocol.get("action_reference", "direct") != "direct"):
                    raise ValueError("Fixed benchmark configuration changed")
                if canonical_classical is None:
                    canonical_classical = values
                if any(not np.array_equal(values[k], canonical_classical[k]) for k in values):
                    raise ValueError("Repeated fixed benchmark differs across seeds")
            else:
                for key, default in (("algorithm", None), ("guidance", False), ("filter", False), ("static_filter", False),
                                     ("action_reference", "direct"), ("conflict_features", False),
                                     ("mask_conflict_features", False)):
                    if protocol.get(key, default) != config.get(key, default):
                        raise ValueError("Evaluation changed " + key)
                selected, record = select_checkpoint(training, stage)
                if protocol["checkpoint"] != record:
                    raise ValueError("Evaluation did not use its own registered checkpoint")
                checkpoint = verified_checkpoint(training, record)
                if checkpoint != selected.resolve():
                    raise ValueError("Checkpoint selection and verification disagree")
                if stage == "final" and record["sha256"] != checked["model_sha256"]:
                    raise ValueError("Final evaluation differs from the audited policy")
                recorded_points.append({
                    "stage": stage, "checkpoint": record,
                    "original_model_path": protocol["model_path"],
                    "verified_local_model_path": str(checkpoint),
                    "means": {k: float(v.mean()) for k, v in values.items()},
                    "csv_sha256": summary["csv_sha256"],
                })
            evaluated[stage] = values

        means = {stage: {k: float(v.mean()) for k, v in values.items()}
                 for stage, values in evaluated.items()}
        learning = {metric: paired_effect(values, evaluated["initial"][metric], indices)
                    for metric, values in evaluated["final"].items()}
        versus_classical = {metric: paired_effect(values, evaluated["classical"][metric], indices)
                            for metric, values in evaluated["final"].items()}
        checks = screen(means["initial"], means["final"], plan["advance_screen"])
        rows.append({
            "seed": seed, "audit": checked, "means": means, "points": recorded_points,
            "learning_effects": learning, "trained_minus_classical": versus_classical,
            "registered_screen_checks": checks, "screen_passed": all(checks.values()),
            "training_config_sha256": digest(training / "config.json"),
            "training_provenance_sha256": digest(training / "provenance.json"),
        })

    return {
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seeds": seeds, "all_registered_seeds_included": True,
        "evaluation_worlds": worlds, "unseen_scenarios_used": False,
        "per_seed": rows, "aggregate": aggregate_effects(rows, seeds),
        "classical_means": {k: float(v.mean()) for k, v in canonical_classical.items()},
        "disjoint_training_world_seeds": sorted(used_world_seeds),
        "training_environment": canonical_environment,
        "evaluation_source_sha256": expected_evaluation,
        "classical_source_sha256": expected_classical,
        "classical_uses_same_evaluator_source": expected_classical == expected_evaluation,
        "analysis_script_sha256": digest(Path(__file__)),
        "limitations": [
            "Development worlds; this is not held-out generalization evidence.",
            "Within-seed intervals resample whole worlds, not individual aircraft.",
            "Sample standard deviations describe the registered training seeds; three seeds provide limited precision.",
            "Initial-versus-trained comparisons isolate learning within this supported action representation.",
            "No official scalar competition score or cross-algorithm ranking is inferred.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new analysis output file")
    source = args.source_root.resolve()
    plan_path = source / "jobs/ppo_baseline_v1.json"
    plan = read(plan_path)
    trials = []
    for seed in plan["seeds"]:
        folder = args.results.resolve() / f"seed-{seed}"
        trials.append({
            "seed": seed, "training": folder / "train",
            "classical": folder / "evaluations/classical",
            "evaluations": {stage: folder / "evaluations" / stage for stage in plan["evaluation"]["stages"]},
        })
    result = analyze(plan, trials, source, source)
    result["registered_protocol_sha256"] = digest(plan_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"results": str(args.out.resolve()), "aggregate": result["aggregate"],
                     "screen_passed": {row["seed"]: row["screen_passed"] for row in result["per_seed"]}}))


if __name__ == "__main__":
    main()
