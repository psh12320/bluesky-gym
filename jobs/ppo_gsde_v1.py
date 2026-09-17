"""Run the registered PPO exploration-persistence study using existing training and evaluation modules."""
import argparse
import hashlib
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
    plan = read(ROOT / "jobs/ppo_gsde_v1.json")
    row = plan["rows"][args.index]
    seed, arm = row["seed"], row["arm"]
    parent = ROOT / "runs/cluster-ppo-gsde-v1" / f"seed-{seed}" / arm
    training = parent / "train"
    expected = {**plan["config"], **plan["arms"][arm], "seed": seed}
    if args.action == "train":
        parent.mkdir(parents=True, exist_ok=False)
        with (parent/'launch.json').open('x',encoding='utf-8') as stream:
            json.dump({'protocol_sha256':hashlib.sha256((ROOT/'jobs/ppo_gsde_v1.json').read_bytes()).hexdigest(),
                       'row_index':args.index,'settings':expected},stream,indent=2)
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
    if read(parent/'launch.json')['protocol_sha256'] != hashlib.sha256((ROOT/'jobs/ppo_gsde_v1.json').read_bytes()).hexdigest():
        raise ValueError('Protocol changed after training launch')
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
    if args.action == 'train':
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from atc_rl.exploration import validate_model
        restored=PPO.load(training/'model.zip',device='cuda')
        validate_model(restored,config)
        assert restored.device.type=='cuda' and next(restored.policy.parameters()).device.type=='cuda'
        observation={key:np.zeros(space.shape,dtype=np.float32) for key,space in restored.observation_space.spaces.items()}
        observation['actor'][-1]=1.0
        action=restored.predict(observation,deterministic=True)[0]
        assert action.shape==(2,) and np.isfinite(action).all() and (np.abs(action)<=1).all()
        result={'device':str(restored.device),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
                'cuda_runtime':torch.version.cuda,'sde_sample_freq':restored.sde_sample_freq,
                'bounded_finite_action':True,'model_sha256':summary['model_sha256']}
        with (parent/'cuda-reload.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
        del restored
        if args.index==0:
            import csv
            from atc.metrics import METRICS
            run('atc_rl.evaluate','--model',training/'model.zip','--episodes',2,'--seed',20260,'--out',parent/'native-check-vector')
            run('atc_rl.competition','--env','ma','--model',training/'model.zip','--episodes',2,'--seed',20260,'--out',parent/'native-check-original')
            def records(directory):
                with (directory/'aircraft.csv').open(newline='',encoding='utf-8') as stream:
                    rows=list(csv.DictReader(stream))
                keyed={(int(r['episode']),r['agent']):r for r in rows}
                assert len(rows)==len(keyed)==20
                return keyed
            left=records(parent/'native-check-vector');right=records(parent/'native-check-original')
            assert left.keys()==right.keys()
            for key in left:
                assert left[key]['scenario_sha256']==right[key]['scenario_sha256']
                for metric in METRICS:assert float(left[key][metric])==float(right[key][metric]),(key,metric)
            with (parent/'native-check.json').open('x',encoding='utf-8') as stream:
                json.dump({'matched_metric_values':180,'model_sha256':summary['model_sha256'],
                           'development_worlds':2,'performance_evidence':False},stream,indent=2)
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
    marker={'completed':args.action,'seed':seed,'arm':arm,'live_transitions':summary['live_transitions'],
            'protocol_sha256':hashlib.sha256((ROOT/'jobs/ppo_gsde_v1.json').read_bytes()).hexdigest()}
    with (parent/f'{args.action}-complete.json').open('x',encoding='utf-8') as stream:json.dump(marker,stream,indent=2)
    print(json.dumps({"completed": args.action, "seed": seed, "arm": arm, "live_transitions": summary["live_transitions"]}), flush=True)

if __name__ == "__main__":
    main()
