"""Fit the registered supervised start, validate PPO continuation, then run the RL pilot."""
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
    if sha(Path(plan["bundle"])) != plan["bundle_sha256"]:
        raise ValueError("Frozen pipeline source changed")
    (OUT / "started").mkdir(exist_ok=False)
    original_protocol, original_script = sha(OUT / "protocol.json"), sha(Path(__file__))
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    def execute(args, log):
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
        values = {**settings, "run_dir": folder / "train", "pretrained_model": OUT / "bc/model.zip"}
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
        if policy_fingerprint(folder / "train/initial-model.zip") != policy_fingerprint(OUT / "bc/model.zip"):
            raise ValueError("PPO starting policy changed after supervised fitting")
        return folder

    print(json.dumps({"stage": "waiting_for_registered_training_and_validation_data"}), flush=True)
    wait_started = time.monotonic()
    while not all((Path(plan[key]) / "complete.json").is_file() for key in ("training_data", "validation_data")):
        if time.monotonic() - wait_started > plan["dependency_timeout_seconds"]:
            raise TimeoutError("Dataset dependency did not complete; no training was launched")
        time.sleep(5)
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
    execute(command("atc_rl.imitation", {
        **plan["imitation"], "training_data": plan["training_data"],
        "validation_data": plan["validation_data"], "out": OUT / "bc",
    }), OUT / "bc.log")
    bc = read(OUT / "bc/training_summary.json")
    if bc["status"] != "complete" or bc["reinforcement_learning_updates"] != 0 or not bc["serialization_actions_identical"]:
        raise ValueError("Supervised fitting did not pass its technical checks")

    technical = train("technical", plan["technical"])
    execute(command("atc_rl.evaluate", {"model": technical / "train/model.zip", "episodes": 2,
            "seed": 20260, "out": technical / "vector"}), technical / "vector.log")
    execute(command("atc_rl.competition", {"env": "ma", "model": technical / "train/model.zip", "episodes": 2,
            "seed": 20260, "out": technical / "native"}), technical / "native.log")
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
        "status": "complete", "exact_physical_metric_matches": 180,
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
