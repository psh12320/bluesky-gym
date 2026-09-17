"""Run the registered static-support study using existing training and evaluation modules."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atc_rl.cluster import verify
from atc_rl.matrix_evaluate import select_checkpoint

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def run(module, *args):
    command = [sys.executable, "-u", "-m", module, *map(str, args)]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--index", type=int, choices=range(6), required=True)
    args = parser.parse_args()
    verify(ROOT)
    plan = read(ROOT / "jobs/ppo_static_v1.json")
    row = plan["rows"][args.index]
    seed, arm = row["seed"], row["arm"]
    parent = ROOT / "runs/cluster-ppo-static-v1" / f"seed-{seed}" / arm
    training = parent / "train"
    expected = {**plan["config"], **plan["arms"][arm], "seed": seed}
    if args.action == "train":
        parent.mkdir(parents=True, exist_ok=False)
        if args.index == 0:
            run("pytest", "tests/rl", "-q", "-p", "no:cacheprovider",
                "--basetemp", parent / "tests", "--junitxml", parent / "tests.xml")
        command = []
        for key, value in expected.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.extend([flag, str(value)])
        run("atc_rl.train", *command, "--run-dir", training)
        run("atc_rl.audit", "--run", training, "--out", training / "checkpoint-audit.json")
    config = read(training / "config.json")
    summary = read(training / "training_summary.json")
    audit = read(training / "checkpoint-audit.json")
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Training differs from the registered recipe")
    maximum_rollout = expected["workers"] * 10 * expected["rollout_steps"]
    if summary["status"] != "complete" or not expected["live_steps"] <= summary["live_transitions"] < expected["live_steps"] + maximum_rollout:
        raise ValueError("Partial training preserved; requested live-transition budget not completed")
    for key in ("status", "live_transitions", "optimizer_steps", "model_sha256"):
        if audit[key] != summary[key]:
            raise ValueError("Checkpoint audit does not match training summary")
    runtime = read(training / "runtime.json")
    if not runtime["static_area_filter"] or runtime["traffic_conflict_filter"]:
        raise ValueError("Static-only training accidentally changed action support")
    manifest = read(ROOT / "cluster-manifest.json")
    source = {name: digest for name, digest in manifest["files"].items()
              if name == "pyproject.toml" or name.endswith(".py") and name.split("/")[0] in
              ("atc", "atc_rl", "core", "bluesky_gym", "bluesky_zoo")}
    if read(training / "provenance.json")["source_sha256"] != source:
        raise ValueError("Training source differs from the immutable bundle")
    if args.action == "evaluate":
        output = parent / "evaluations"
        output.mkdir(exist_ok=False)
        common = ("--episodes", plan["evaluation"]["worlds"], "--seed", plan["evaluation"]["seed"])
        classical = output / "classical"
        run("atc_rl.evaluate", "--classical", *common, "--out", classical)
        evaluations = {}
        for stage in plan["evaluation"]["stages"]:
            checkpoint, _ = select_checkpoint(training, stage)
            evaluations[stage] = output / stage
            run("atc_rl.evaluate", "--model", checkpoint, *common, "--out", evaluations[stage])
        run("atc_rl.compare", "--initial", evaluations["initial"], "--trained", evaluations["final"],
            "--classical", classical, "--out", parent / "comparison")
        curve_args = []
        for directory in evaluations.values():
            curve_args.extend(["--evaluation", directory])
        run("atc_rl.curves", "--training-run", training, "--classical", classical,
            "--out", parent / "curve", *curve_args)
    verify(ROOT)
    print(json.dumps({"completed": args.action, "seed": seed, "arm": arm, "live_transitions": summary["live_transitions"]}), flush=True)

if __name__ == "__main__":
    main()
