"""Reuse the preserved supervised policy, validate the corrected PPO handoff, and run the registered pilot."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
read = lambda path: json.loads(Path(path).read_text(encoding="utf-8-sig"))
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(module, values):
    args = [sys.executable, "-u", "-m", module]
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                args.append("--" + key.replace("_", "-"))
        else:
            args.extend(["--" + key.replace("_", "-"), str(value)])
    return args


def main():
    plan = read(OUT / "protocol.json")
    source = Path(plan["source"])
    sys.path.insert(0, str(source))
    from atc_rl.cluster import verify
    from atc_rl.demonstrations import load_demonstrations, require_disjoint
    from atc_rl.checkpoint_identity import verified_checkpoint, policy_fingerprint
    from atc.metrics import METRICS
    verify(source)
    for name, expected in plan["protected_original_files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Original experiment artifact changed: " + name)
    if sha(Path(plan["bundle"])) != plan["bundle_sha256"]:
        raise ValueError("Frozen pipeline source changed")
    (OUT / "started").mkdir(exist_ok=False)
    original_protocol, original_script = sha(OUT / "protocol.json"), sha(Path(__file__))
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    def execute(args, log):
        for name, expected in plan["protected_original_files"].items():
            if sha(ROOT / name) != expected:
                raise ValueError("Original experiment artifact changed: " + name)
        if sha(OUT / "protocol.json") != original_protocol or sha(Path(__file__)) != original_script:
            raise ValueError("Running experiment plan changed")
        print(json.dumps({"stage": str(log.relative_to(OUT)), "at_utc": datetime.now(timezone.utc).isoformat()}), flush=True)
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(args, cwd=source, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Stage failed; retain " + str(log))
    def train(name, settings):
        folder = OUT / name
        folder.mkdir()
        values = {**settings, "run_dir": folder / "train", "pretrained_model": Path(plan["pretrained_model"])}
        args = command("atc_rl.train", values)
        (folder / "launch.json").write_text(json.dumps({"command": args, "protocol_sha256": original_protocol}, indent=2))
        execute(args, folder / "train.log")
        summary = read(folder / "train/training_summary.json")
        maximum = settings["workers"] * 10 * settings["rollout_steps"]
        if summary["status"] != "complete" or not settings["live_steps"] <= summary["live_transitions"] < settings["live_steps"] + maximum:
            raise ValueError("Incomplete or incorrect PPO budget")
        execute(command("atc_rl.audit", {"run": folder / "train", "out": folder / "audit.json"}), folder / "audit.log")
        audited = read(folder / "audit.json")
        if audited["initial_policy_is_untrained"] or not audited["pretraining"]:
            raise ValueError("PPO audit lost the supervised history")
        if policy_fingerprint(folder / "train/initial-model.zip") != policy_fingerprint(Path(plan["pretrained_model"])):
            raise ValueError("PPO starting policy changed after supervised fitting")
        return folder

    training = load_demonstrations(plan["training_data"], "train")
    validation = load_demonstrations(plan["validation_data"], "validation")
    require_disjoint(training, validation)
    data_plan = read(Path(plan["data_protocol"]))
    if sha(Path(plan["data_protocol"])) != plan["data_protocol_sha256"]:
        raise ValueError("Demonstration collection plan changed")
    for name, data in (("training", training), ("validation", validation)):
        for key in ("seed", "worlds", "role"):
            if data["protocol"][key] != data_plan[name][key]:
                raise ValueError("Dataset differs from registered collection")
    (OUT / "data-check.json").write_text(json.dumps({
        "training_transitions": len(training["arrays"]["actor"]),
        "validation_transitions": len(validation["arrays"]["actor"]),
        "training_worlds": len(training["scenarios"]), "validation_worlds": len(validation["scenarios"]),
        "scenarios_disjoint": True, "evaluation_data_used": False,
        "training_complete_sha256": sha(Path(plan["training_data"]) / "complete.json"),
        "validation_complete_sha256": sha(Path(plan["validation_data"]) / "complete.json"),
    }, indent=2))
    from atc_rl.pretrained import inspect_parent
    _, bc, parent_record = inspect_parent(plan["pretrained_model"])
    if bc["status"] != "complete" or bc["reinforcement_learning_updates"] != 0 or not bc["serialization_actions_identical"]:
        raise ValueError("Preserved supervised fitting did not pass its technical checks")
    (OUT / "reused-parent.json").write_text(json.dumps({
        "original_model": plan["pretrained_model"], "model_sha256": sha(Path(plan["pretrained_model"])),
        "parent_checkpoint": parent_record, "summary": bc,
        "new_supervised_updates": 0, "original_attempt_preserved": True,
    }, indent=2), encoding="utf-8")

    technical = OUT / "technical"
    technical.mkdir()
    reference = Path(plan["reused_technical_training"])
    summary = read(reference / "training_summary.json")
    if summary["status"] != "complete" or summary["training_seed"] != plan["technical"]["seed"]:
        raise ValueError("Registered technical training is incomplete or belongs to another seed")
    if sha(reference / "model.zip") != plan["action_precision_recovery"]["technical_model_sha256"]:
        raise ValueError("Technical checkpoint changed")
    if policy_fingerprint(reference / "initial-model.zip") != policy_fingerprint(Path(plan["pretrained_model"])):
        raise ValueError("Reused technical training did not start from the preserved BC policy")
    execute(command("atc_rl.audit", {"run": reference, "out": technical / "audit.json"}), technical / "audit.log")
    (technical / "reused-training.json").write_text(json.dumps({
        "directory": str(reference), "model_sha256": sha(reference / "model.zip"),
        "summary": summary, "new_technical_rl_updates": 0,
    }, indent=2), encoding="utf-8")
    execute(command("atc_rl.evaluate", {"model": reference / "model.zip", "episodes": 2,
            "seed": 20260, "out": technical / "vector"}), technical / "vector.log")
    execute(command("atc_rl.competition", {"env": "ma", "model": reference / "model.zip", "episodes": 2,
            "seed": 20260, "out": technical / "native"}), technical / "native.log")
    if sha(technical / "vector/aircraft.csv") != plan["original_vector_csv_sha256"]:
        raise ValueError("Corrected source changed vector evaluation results")
    def rows(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return {(int(row["episode"]), row["agent"]): row for row in csv.DictReader(stream)}
    vector, native = rows(technical / "vector/aircraft.csv"), rows(technical / "native/aircraft.csv")
    if vector.keys() != native.keys() or len(vector) != 20:
        raise ValueError("Incomplete native/vector technical evaluation")
    for key in vector:
        if vector[key]["scenario_sha256"] != native[key]["scenario_sha256"]:
            raise ValueError("Technical evaluation scenarios differ")
        for metric in METRICS:
            if float(vector[key][metric]) != float(native[key][metric]):
                raise ValueError("Native/vector mismatch in " + metric)
    (technical / "complete.json").write_text(json.dumps({
        "status": "complete", "exact_metric_matches": 180, "exact_physical_metric_matches": 160, "exact_reward_matches": 20,
        "technical_only": True, "used_for_model_selection": False,
    }, indent=2))

    pilot = train("ppo", plan["ppo"])
    records = read(pilot / "train/checkpoints.json")
    middle = min((r for r in records if r["file"].startswith("policy-live-") and r["live_transitions"] >= 50000),
                 key=lambda r: r["live_transitions"])
    if middle["live_transitions"] >= 50000 + plan["ppo"]["workers"] * 10 * plan["ppo"]["rollout_steps"]:
        raise ValueError("Intermediate checkpoint is outside the registered tolerance")
    selected = {
        "initial": next(r for r in records if r["file"] == "initial-model.zip"),
        "50k": middle, "final": next(r for r in records if r["file"] == "model.zip"),
    }
    for stage, record in selected.items():
        model = verified_checkpoint(pilot / "train", record)
        execute(command("atc_rl.evaluate", {"model": model, "episodes": 20, "seed": 20260,
                "out": pilot / f"eval-{stage}-dev20"}), pilot / f"eval-{stage}.log")
    execute(command("atc_rl.compare", {
        "initial": pilot / "eval-initial-dev20", "trained": pilot / "eval-final-dev20",
        "classical": plan["classical"], "out": pilot / "comparison",
    }), pilot / "comparison.log")
    args = command("atc_rl.curves", {"training_run": pilot / "train", "classical": plan["classical"], "out": pilot / "curve"})
    for stage in selected:
        args.extend(["--evaluation", str(pilot / f"eval-{stage}-dev20")])
    execute(args, pilot / "curve.log")
    verify(source)
    for name, expected in plan["protected_original_files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Original experiment artifact changed: " + name)
    (OUT / "complete.json").write_text(json.dumps({
        "status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": original_protocol, "supervised_training": bc,
        "ppo_training": read(pilot / "train/training_summary.json"),
        "learning_reference": "The PPO initial checkpoint is the verified behavior-cloned policy, not an untrained actor.",
        "candidate_promoted": False, "unseen_scenarios_used": False,
    }, indent=2))
    print(json.dumps({"status": "complete", "review_required": True}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        with (OUT / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump({"error": repr(error), "all_outputs_retained": True}, stream, indent=2)
        raise
