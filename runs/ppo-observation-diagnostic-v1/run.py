"""Inspect inputs and frozen actor outputs along one unchanged reference trajectory."""
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
sys.path.insert(0, str(SOURCE))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from atc.conflicts import FEATURE_NAMES
from atc_rl.cluster import verify
from atc_rl.world_pool import WorldPool

read = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    verify(SOURCE)
    models = {
        "guided_49900_100k": ROOT / "runs/ppo-guidance-pilot-v1/train/model.zip",
        "guided_49920_100k": ROOT / "runs/ppo-guidance-replication-v1/seed-49920/train/model.zip",
        "guided_49940_100k": ROOT / "runs/ppo-guidance-replication-v1/seed-49940/train/model.zip",
        "features_50400_100k": ROOT / "runs/ppo-cpa-matched-v1/seed-50400/features/train/model.zip",
    }
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE), "source_bundle_sha256": sha(SOURCE.with_name(SOURCE.name.removesuffix("-verify") + ".zip")),
        "worlds": 1, "stream": 20260, "behavior": "Zero latent actions, goal_offset, guidance on, filter off.",
        "sample": "Every live aircraft observation before every fifth world decision.",
        "purpose": "Inspect input scales, hidden saturation and action variation on a fixed reference trajectory.",
        "models": {name: {"path": str(path), "sha256": sha(path)} for name, path in models.items()},
        "performance_comparison": False,
        "limitations": [
            "One development world under the untrained reference, not each learned policy's state distribution.",
            "Actor outputs are inspected without commanding the simulator.",
            "Input scales and activation summaries alone cannot establish the cause of failed learning.",
            "Different architectures and training seeds are descriptive here, not a controlled algorithm ranking.",
        ],
    }
    with (OUT / "protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2)
    environment = WorldPool(1, OUT / "workers", guidance=True, filter=False,
                            action_reference="goal_offset", conflict_features=True)
    observations, timestamps, aircraft_ids, records = [], [], [], []
    try:
        environment.seed(20260)
        observation = environment.reset()
        scenario = environment.reset_infos[0]["scenario_sha256"]
        dictionary = environment.get_attr("observation_spaces", indices=[0])[0]["KL001"]
        layout, cursor = {}, 0
        for name, space in dictionary.spaces.items():
            size = spaces.flatdim(space)
            layout[name] = [cursor, cursor + size]
            cursor += size
        layout["time_remaining"] = [cursor, cursor + 1]
        assert cursor + 1 == observation["actor"].shape[1] == 170
        live = np.ones(10, dtype=bool)
        decision, start = 0, time.monotonic()
        while True:
            if decision % 5 == 0:
                observations.append(observation["actor"][live].copy())
                timestamps.extend([decision * 5] * int(live.sum()))
                aircraft_ids.extend(np.flatnonzero(live).tolist())
            observation, _, dones, infos = environment.step(np.zeros((10, 2), dtype=np.float32))
            for index, info in enumerate(infos):
                if info["aircraft_done"]:
                    records.append({"episode": 0, "agent": f"KL00{index+1}",
                                    "scenario_sha256": scenario, **info["metrics"]})
            decision += 1
            if infos[0]["world_completed"]:
                break
            live &= ~dones
            if decision > 600 or time.monotonic() - start > 600:
                raise RuntimeError("Diagnostic exceeded its one-world bound; preserve outputs")
        elapsed = time.monotonic() - start
    finally:
        environment.close()
    assert len(records) == 10
    reference_path = ROOT / "runs/initial-support-ablation-v1/eval-g1-f0-dev20/aircraft.csv"
    with reference_path.open(newline="", encoding="utf-8") as stream:
        reference = {row["agent"]: row for row in csv.DictReader(stream) if row["episode"] == "0"}
    from atc.metrics import METRICS
    for row in records:
        expected = reference[row["agent"]]
        assert row["scenario_sha256"] == expected["scenario_sha256"]
        for metric in METRICS:
            assert row[metric] == float(expected[metric]), (row["agent"], metric)
    samples = np.concatenate(observations)
    np.savez_compressed(OUT / "observations.npz", observations=samples,
                        seconds=np.asarray(timestamps), aircraft=np.asarray(aircraft_ids))
    with (OUT / "layout.json").open("x", encoding="utf-8") as stream:
        json.dump(layout, stream, indent=2)
    field_stats = {}
    for name, (start, stop) in layout.items():
        values = samples[:, start:stop]
        field_stats[name] = {
            "rms": float(np.sqrt(np.mean(values ** 2))),
            "max_absolute": float(np.abs(values).max()),
            "nonzero_fraction": float(np.mean(values != 0)),
            "dimension": stop - start,
        }
    keep = np.array([i for name, (start, stop) in layout.items() if name not in FEATURE_NAMES
                     for i in range(start, stop)])
    assert len(keep) == 125
    model_stats = {}
    for name, path in models.items():
        assert sha(path) == protocol["models"][name]["sha256"]
        model = PPO.load(path, device="cpu")
        config = read(path.parent / "config.json")
        values = samples if config.get("conflict_features", False) else samples[:, keep]
        assert values.shape[1] == model.observation_space["actor"].shape[0]
        actor = torch.as_tensor(values, dtype=torch.float32)
        network = model.policy.mlp_extractor.policy_net
        with torch.no_grad():
            first = network[1](network[0](actor))
            second = network[3](network[2](first))
            means = model.policy.action_net(second).numpy()
            log_std = model.policy.log_std.numpy()
        model_stats[name] = {
            "checkpoint_sha256": sha(path),
            "live_transitions": read(path.parent / "training_summary.json")["live_transitions"],
            "actor_tanh1_fraction_abs_above_0_99": float((first.abs() > .99).float().mean()),
            "actor_tanh2_fraction_abs_above_0_99": float((second.abs() > .99).float().mean()),
            "learned_stochastic_standard_deviation": np.exp(log_std).tolist(),
            "latent_mean_by_action": means.mean(axis=0).tolist(),
            "latent_std_across_reference_states": means.std(axis=0).tolist(),
            "latent_5_50_95_percentiles_by_action": np.quantile(means, [.05, .5, .95], axis=0).tolist(),
            "fraction_mean_outside_action_bounds": float(np.mean(np.abs(means) > 1)),
            "heading_offset_degrees_mean": float(means[:, 0].clip(-1, 1).mean() * 90),
            "heading_offset_degrees_std": float(means[:, 0].clip(-1, 1).std() * 90),
        }
        del model
    result = {
        "sampled_live_observations": len(samples), "world_decisions": decision,
        "reference_world_seconds": elapsed, "scenario_sha256": scenario,
        "reference_physical_metrics_matched": 90,
        "reference_csv_sha256": sha(reference_path),
        "input_fields": field_stats, "models": model_stats,
        "limitations": protocol["limitations"],
    }
    with (OUT / "results.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"samples": len(samples), "reference_metrics_matched": 90, "models": model_stats}), flush=True)


if __name__ == "__main__":
    main()
