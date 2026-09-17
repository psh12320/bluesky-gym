"""Check the original rollout loops on development scenarios, without using seed 42."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from atc.compare import load_evaluation
from atc.provenance import capture
from scripts import evaluate_competition as harness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Existing development evaluation prefix")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata, expected = load_evaluation(args.reference)
    if metadata["seed"] != 2026:
        parser.error("Integration checks use only development seed 2026")
    if not 1 <= args.episodes <= metadata["episodes"]:
        parser.error("Requested episode prefix is unavailable in the reference")
    if args.out.exists():
        parser.error("Check output already exists")
    if args.model and not args.model.is_file():
        parser.error("Checkpoint must be an existing ZIP file")
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest() if args.model else None
    if model_hash != metadata.get("model_sha256"):
        parser.error("Checkpoint hash differs from the reference evaluation")
    if not args.model and (metadata.get("recipe", "baseline") != "baseline"
                           or metadata.get("guard_static", False)
                           or metadata.get("policy", "neutral") != "neutral"):
        parser.error("Model-free harness checks use the original unfiltered neutral baseline")
    capture(args.out.parent / (args.out.stem + "_provenance"))
    # Only this development process overrides the module variable. The source
    # harness and the actual reported protocol retain SEED=42 and 1000 episodes.
    harness.SEED = 2026
    kind = metadata["env"]
    act = harness.load_policy(kind, args.model)
    if kind == "sa":
        actual = harness.run_single_agent(args.episodes, act)
    else:
        actual = harness.run_multi_agent(args.episodes, act, 10)
    expected = [row for row in expected if row["episode"] < args.episodes]
    if len(actual) != len(expected):
        raise AssertionError(f"Aircraft count differs: {len(actual)} versus {len(expected)}")
    differences = {}
    aggregate_differences = {}
    for key in harness.METRIC_KEYS:
        observed = np.array([row[key] for row in actual])
        reference = np.array([row[key] for row in expected])
        differences[key] = float(np.max(np.abs(observed - reference)))
        aggregate_differences[key] = float(observed.mean() - reference.mean())
    passed = all(delta <= (1e-5 if key == "total_reward" else 0.0)
                 for key, delta in differences.items())
    result = {"env": kind, "seed": 2026, "episodes": args.episodes,
              "agent_episodes": len(actual), "official_protocol": False,
              "model_sha256": model_hash, "reference": str(args.reference),
              "matches_reference": passed, "max_absolute_metric_differences": differences,
              "aggregate_metric_differences": aggregate_differences,
              "actual_records": [{key: float(row[key]) for key in harness.METRIC_KEYS} for row in actual]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "actual_records"}, indent=2))
    if not passed:
        raise AssertionError("Original harness differs from the development reference; inspect the saved differences")


if __name__ == "__main__":
    main()
