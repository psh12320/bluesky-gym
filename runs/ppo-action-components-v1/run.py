"""A bounded action-component diagnostic; no model or controller is retrained."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/onpolicy-cpa-source-v2-verify"
TRAINING = ROOT / "runs/ppo-cpa-matched-v1/seed-50400/features/train"
sys.path.insert(0, str(SOURCE))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import torch
from stable_baselines3 import PPO
from atc.metrics import METRICS, summarize
from atc_rl.cluster import verify
from atc_rl.checkpoint_identity import verified_checkpoint
from atc_rl.world_pool import WorldPool

read = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def first_two(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["episode"] in ("0", "1")]
    assert len(rows) == 20
    return {(int(row["episode"]), row["agent"]): row for row in rows}


def main():
    torch.set_num_threads(1)
    verify(SOURCE)
    record = next(r for r in read(TRAINING / "checkpoints.json") if r["file"] == "model.zip")
    checkpoint = verified_checkpoint(TRAINING, record)
    model = PPO.load(checkpoint, device="cpu")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": record, "source": str(SOURCE), "diagnostic_script_sha256": sha(Path(__file__)),
        "worlds_per_arm": 2, "stream": 20260,
        "arms": ["full_policy", "zero_heading", "zero_speed"],
        "mapping": "Zero the selected latent action after deterministic model prediction, before goal_offset mapping.",
        "support": {"guidance": True, "filter": False, "conflict_features": True},
        "reason": "A passive reference-trajectory probe found a mean heading offset near -7.2 degrees for this policy.",
        "selection": "First completed real-feature pilot; selected before these component evaluations.",
        "limits": [
            "Two development worlds; diagnostic evidence only.",
            "Heading and speed interact through subsequent states; this is a closed-loop policy-component ablation.",
            "Changing inference behavior is not additional RL training.",
            "These variants are not nominated competition candidates.",
        ],
    }
    with (OUT / "protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2)
    all_records, results, scenario_sequences = {}, {}, {}
    for arm in protocol["arms"]:
        folder = OUT / arm
        folder.mkdir(exist_ok=False)
        env = WorldPool(1, folder / "workers", guidance=True, filter=False,
                        action_reference="goal_offset", conflict_features=True)
        records, scenarios = [], []
        start = time.monotonic()
        try:
            assert env.observation_space == model.observation_space
            env.seed(20260)
            observation = env.reset()
            scenario = env.reset_infos[0]["scenario_sha256"]
            episode = decisions = 0
            with (folder / "aircraft.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["episode", "scenario_sha256", "agent", *METRICS])
                writer.writeheader()
                while episode < 2:
                    actions = []
                    for index in range(10):
                        local = {key: observation[key][index] for key in observation}
                        local["critic"] = np.zeros_like(local["critic"])
                        actions.append(model.predict(local, deterministic=True)[0])
                    actions = np.asarray(actions, dtype=np.float32)
                    if arm == "zero_heading":
                        actions[:, 0] = 0
                    elif arm == "zero_speed":
                        actions[:, 1] = 0
                    observation, _, _, infos = env.step(actions)
                    decisions += 1
                    for index, info in enumerate(infos):
                        if info["aircraft_done"]:
                            row = {"episode": episode, "scenario_sha256": scenario,
                                   "agent": f"KL00{index+1}", **info["metrics"]}
                            writer.writerow(row)
                            stream.flush()
                            records.append(row)
                    if infos[0]["world_completed"]:
                        scenarios.append(scenario)
                        episode += 1
                        scenario = env.reset_infos[0]["scenario_sha256"]
                    if decisions > 1200 or time.monotonic() - start > 300:
                        raise RuntimeError("Diagnostic exceeded its two-world bound; retain outputs")
        finally:
            env.close()
        result = summarize(records, 2, 10)
        result.update(arm=arm, checkpoint_sha256=record["sha256"],
                      scenarios=scenarios, world_decisions=decisions,
                      csv_sha256=sha(folder / "aircraft.csv"),
                      wall_seconds=time.monotonic() - start, diagnostic_only=True)
        with (folder / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        all_records[arm] = records
        results[arm] = result
        scenario_sequences[arm] = scenarios
        print(json.dumps({"arm_completed": arm, "clean_completion": result["clean_completion_rate"],
                          "metrics": {k: v["mean"] for k, v in result["metrics"].items()}}), flush=True)
    assert len({tuple(v) for v in scenario_sequences.values()}) == 1
    existing_path = TRAINING.parent / "eval-final-dev20/aircraft.csv"
    existing = first_two(existing_path)
    for row in all_records["full_policy"]:
        expected = existing[row["episode"], row["agent"]]
        assert row["scenario_sha256"] == expected["scenario_sha256"]
        for metric in METRICS:
            assert row[metric] == float(expected[metric]), (row["episode"], row["agent"], metric)
    initial_path = TRAINING.parent / "eval-initial-dev20/aircraft.csv"
    initial_raw = first_two(initial_path)
    initial = [{**row, "episode": episode, **{metric: float(row[metric]) for metric in METRICS}}
               for (episode, _), row in initial_raw.items()]
    results["own_initial_policy"] = summarize(initial, 2, 10)
    assert sha(checkpoint) == record["sha256"]
    verify(SOURCE)
    with (OUT / "results.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "purpose": "Small action-component diagnostic; no performance selection.",
            "full_policy_metrics_matched_to_existing_evaluation": 180,
            "all_arms_use_identical_scenarios": True, "results": results,
            "limitations": protocol["limits"],
        }, stream, indent=2)


if __name__ == "__main__":
    main()
