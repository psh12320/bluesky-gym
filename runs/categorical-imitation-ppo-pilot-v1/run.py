"""Measure navigation-prior, imitation, and PPO contributions with fixed categorical commands."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def main():
    plan = read(OUT / "protocol.json")
    source = Path(plan["source"])
    sys.path.insert(0, str(source))
    from atc_rl.cluster import verify
    from atc_rl.checkpoint_identity import policy_fingerprint, verified_checkpoint
    from atc_rl.compare import load_evaluation
    from atc_rl.demonstrations import load_demonstrations, require_disjoint
    from atc_rl.imitation import categorical_labels, verify_teacher_dependencies
    from atc_rl.maneuvers import ManeuverMapping
    from atc.metrics import METRICS
    import numpy as np

    protocol_sha, script_sha = sha(OUT / "protocol.json"), sha(__file__)

    def preserve():
        verify(source)
        if sha(plan["source_bundle"]) != plan["source_bundle_sha256"]:
            raise ValueError("Registered source archive changed")
        if sha(plan["static_navigation_cache"]) != plan["static_navigation_cache_sha256"]:
            raise ValueError("Registered navigation cache changed")
        for name, digest in plan["protected_files"].items():
            if sha(name) != digest:
                raise ValueError("Protected artifact changed: " + name)
        if sha(OUT / "protocol.json") != protocol_sha or sha(__file__) != script_sha:
            raise ValueError("Running study plan or runner changed")

    preserve()
    (OUT / "started").mkdir(exist_ok=False)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

    def execute(module, values, log):
        preserve()
        command = [sys.executable, "-u", "-m", module]
        for key, value in values.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    command.extend([flag, str(item)])
            else:
                command.extend([flag, str(value)])
        write(log.with_suffix(".launch.json"), {"command": command, "cwd": str(source),
              "protocol_sha256": protocol_sha, "script_sha256": script_sha})
        print(json.dumps({"stage": log.name, "at_utc": datetime.now(timezone.utc).isoformat()}), flush=True)
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=source, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Stage failed ({result.returncode}); retain {log}")
        preserve()

    def parallel(jobs):
        errors = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = {pool.submit(execute, module, values, log): log for module, values, log in jobs}
            for task in as_completed(tasks):
                try:
                    task.result()
                except BaseException as error:
                    errors.append({"stage": str(tasks[task]), "error": repr(error)})
        if errors:
            raise RuntimeError(json.dumps({"failed_stages": errors, "other_started_stages_allowed_to_finish": True}))

    training = load_demonstrations(plan["training_data"], "train")
    validation = load_demonstrations(plan["validation_data"], "validation")
    require_disjoint(training, validation)
    mapping = ManeuverMapping.from_schema(read(Path(plan["training_data"]) / "input-schema.json"))
    if mapping != ManeuverMapping.from_schema(read(Path(plan["validation_data"]) / "input-schema.json")):
        raise ValueError("Train/validation field layouts differ")
    coverage = {label: len(categorical_labels(data, mapping))
                for label, data in (("train", training), ("validation", validation))}
    if coverage != {"train": 62744, "validation": 16108}:
        raise ValueError("Demonstration coverage differs from the registered data")
    dependencies = verify_teacher_dependencies(training, source, plan["teacher_source"])
    write(OUT / "data-preflight.json", {"exactly_reconstructed_rows": coverage,
          "teacher_dependencies": dependencies, "training_validation_worlds_disjoint": True})
    execute("atc_rl.imitation", {**plan["supervised"], "training_data": plan["training_data"],
            "validation_data": plan["validation_data"], "teacher_source": plan["teacher_source"],
            "out": OUT / "bc"}, OUT / "bc.log")
    bc = read(OUT / "bc/training_summary.json")
    if not (bc["status"] == "complete" and bc["epochs"] == 40
            and bc["reinforcement_learning_updates"] == 0
            and bc["supervised_optimizer_steps"] == 2480
            and bc["supervised_examples_seen"] == 2509760
            and bc["serialization_actions_identical"]):
        raise ValueError("Supervised fitting did not meet its registered budget or reload checks")
    bc_records = read(OUT / "bc/checkpoints.json")
    prior_record = next(item for item in bc_records if item["file"] == "untrained-model.zip")
    if prior_record["supervised_optimizer_steps"] != 0 or prior_record["live_transitions"] != 0:
        raise ValueError("Navigation-prior reference already received training")
    prior = verified_checkpoint(OUT / "bc", prior_record)
    parallel([
        ("atc_rl.train", {**plan["ppo"], "pretrained_model": OUT / "bc/model.zip",
                          "run_dir": OUT / "train"}, OUT / "train.log"),
        ("atc_rl.evaluate", {**plan["evaluation"], "model": prior,
                             "out": OUT / "eval-untrained-prior-dev20"}, OUT / "eval-untrained-prior.log"),
    ])
    summary = read(OUT / "train/training_summary.json")
    tolerance = plan["ppo"]["workers"] * 10 * plan["ppo"]["rollout_steps"]
    if summary["status"] != "complete" or not 100000 <= summary["live_transitions"] < 100000 + tolerance:
        raise ValueError("PPO did not reach its registered live-transition budget")
    execute("atc_rl.audit", {"run": OUT / "train", "out": OUT / "audit.json"}, OUT / "audit.log")
    audit = read(OUT / "audit.json")
    if (audit["initial_policy_is_untrained"] or audit["pretraining"]["supervised_optimizer_steps"] != 2480
            or audit["pretraining"]["ppo_optimizer_inherited"]):
        raise ValueError("PPO audit lost supervised lineage or inherited its optimizer")
    if policy_fingerprint(OUT / "bc/model.zip") != policy_fingerprint(OUT / "train/initial-model.zip"):
        raise ValueError("PPO initial tensors differ from the supervised parent")
    records = read(OUT / "train/checkpoints.json")
    middle = min((item for item in records if item["file"].startswith("policy-live-")
                  and item["live_transitions"] >= 50000), key=lambda item: item["live_transitions"])
    if middle["live_transitions"] >= 50000 + tolerance:
        raise ValueError("Intermediate checkpoint exceeded its registered tolerance")
    selected = {"initial": next(item for item in records if item["file"] == "initial-model.zip"),
                "50k": middle, "final": next(item for item in records if item["file"] == "model.zip")}
    write(OUT / "selected-checkpoints.json", selected)
    parallel([("atc_rl.evaluate", {**plan["evaluation"], "model": verified_checkpoint(OUT / "train", record),
                "out": OUT / f"eval-{stage}-dev20"}, OUT / f"eval-{stage}.log")
              for stage, record in selected.items()])
    execute("atc_rl.competition", {**plan["native_check"], "model": OUT / "train/model.zip",
            "out": OUT / "native-final"}, OUT / "native-final.log")

    def rows(path):
        with path.open(newline="", encoding="utf-8") as stream:
            result = list(csv.DictReader(stream))
        indexed = {(int(row["episode"]), row["agent"]): row for row in result}
        if len(indexed) != len(result):
            raise ValueError("Duplicate aircraft rows: " + str(path))
        return indexed

    vector = {key: row for key, row in rows(OUT / "eval-final-dev20/aircraft.csv").items() if key[0] < 2}
    native = rows(OUT / "native-final/aircraft.csv")
    if vector.keys() != native.keys() or len(native) != 20:
        raise ValueError("Native/vector parity check has incomplete aircraft")
    for key in native:
        if native[key]["scenario_sha256"] != vector[key]["scenario_sha256"]:
            raise ValueError("Native/vector scenarios differ")
        for metric in METRICS:
            if float(native[key][metric]) != float(vector[key][metric]):
                raise ValueError("Native/vector mismatch: " + metric)
    write(OUT / "native-check.json", {"exact_metric_matches": 180, "worlds": 2,
          "checkpoint_sha256": sha(OUT / "train/model.zip"), "technical_parity_only": True})
    execute("atc_rl.compare", {"initial": OUT / "eval-initial-dev20", "trained": OUT / "eval-final-dev20",
            "classical": plan["classical"], "out": OUT / "comparison"}, OUT / "comparison.log")
    execute("atc_rl.curves", {"training_run": OUT / "train", "evaluation": [OUT / f"eval-{s}-dev20" for s in selected],
            "classical": plan["classical"], "out": OUT / "curve"}, OUT / "curve.log")
    folders = {"untrained_navigation_prior": OUT / "eval-untrained-prior-dev20",
               "initial_supervised_policy": OUT / "eval-initial-dev20",
               "first_checkpoint_at_least_50k": OUT / "eval-50k-dev20",
               "fixed_final_checkpoint": OUT / "eval-final-dev20",
               "fixed_classical_system": Path(plan["classical"])}
    loaded = {name: load_evaluation(folder) for name, folder in folders.items()}
    initial = loaded["initial_supervised_policy"]
    for name, item in loaded.items():
        if len(rows(folders[name] / "aircraft.csv")) != 200 or len(item[2]) != 20 or item[2] != initial[2]:
            raise ValueError("Incomplete or unpaired performance evaluation: " + name)
        if name != "fixed_classical_system":
            for key in ("source_sha256", "guidance", "filter", "static_filter", "conflict_features",
                        "mask_conflict_features", "action_reference", "exploration", "maneuver_mapping",
                        "evaluation_reward_scale", "evaluation_progress_scale", "traffic_position_scale"):
                if item[1].get(key) != initial[1].get(key):
                    raise ValueError("Contribution comparison changed " + key)
    rng = np.random.default_rng(701)
    indices = rng.integers(0, 20, size=(10000, 20))

    def difference(current, reference):
        result = {}
        for metric, values in loaded[current][3].items():
            delta = values.astype(float) - loaded[reference][3][metric].astype(float)
            result[metric] = {"mean_difference": float(delta.mean()),
                "paired_world_bootstrap_95_percent_interval": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()}
        return result

    means = {name: {key: float(values.mean()) for key, values in item[3].items()} for name, item in loaded.items()}
    contrasts = {
        "supervised_minus_untrained_navigation_prior": difference("initial_supervised_policy", "untrained_navigation_prior"),
        "ppo_final_minus_own_supervised_initial": difference("fixed_final_checkpoint", "initial_supervised_policy"),
        "ppo_final_minus_fixed_classical_system": difference("fixed_final_checkpoint", "fixed_classical_system"),
    }
    final = means["fixed_final_checkpoint"]
    start = means["initial_supervised_policy"]
    screen = plan["advance_screen"]
    changes = contrasts["ppo_final_minus_own_supervised_initial"]
    checks = {
        "arrival_level": final["waypoint_reached"] >= screen["arrival_at_least"],
        "arrival_change": changes["waypoint_reached"]["mean_difference"] >= screen["arrival_change_at_least"] - 1e-12,
        "clean_completion_gain": changes["clean_completion"]["mean_difference"] >= screen["clean_completion_change_at_least"] - 1e-12,
        "flight_time": final["flight_time"] <= start["flight_time"] * (1 + screen["flight_time_relative_increase_at_most"]),
    }
    checks.update({metric: changes[metric]["mean_difference"] <= screen["safety_time_changes_at_most"] + 1e-12
                   for metric in screen["metrics"]})
    preserve()
    write(OUT / "complete.json", {"status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
          "protocol_sha256": protocol_sha, "script_sha256": script_sha, "source_bundle_sha256": plan["source_bundle_sha256"],
          "supervised": bc, "ppo": summary, "audit": audit, "means": means, "contribution_contrasts": contrasts,
          "screen_checks": checks, "passes_replication_screen": all(checks.values()),
          "full_initial_tensors_match_supervised_parent": True, "native_vector_exact_matches": 180,
          "training_seeds": 1, "evaluation_worlds": 20, "unseen_evaluation_performed": False,
          "competitive_performance_claim": False, "limitations": plan["limitations"]})
    print(json.dumps({"status": "complete", "means": means, "screen_checks": checks}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        path = OUT / "failure.json"
        if not path.exists():
            write(path, {"error": repr(error), "artifacts_retained": True,
                         "at_utc": datetime.now(timezone.utc).isoformat()})
        raise
