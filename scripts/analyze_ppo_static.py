"""Audit all six static-support runs and the paired conflict-input contribution."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_rl_cohort import analyze, aggregate_effects, paired_effect, read, digest
from atc_rl.checkpoint_identity import policy_fingerprint
from atc_rl.compare import load_evaluation


def matched_pair(features, zeros):
    """Reject capacity, initialization, runtime or recipe confounds."""
    configs = [read(p / "train/config.json") for p in (features, zeros)]
    ignored = {"run_dir", "mask_conflict_features", "actor_information"}
    if {k: v for k, v in configs[0].items() if k not in ignored} != {
            k: v for k, v in configs[1].items() if k not in ignored}:
        raise ValueError("Paired arms changed more than the input mask")
    if configs[0].get("mask_conflict_features") is not False or configs[1].get("mask_conflict_features") is not True:
        raise ValueError("Paired arms have the wrong feature masks")
    fingerprints = [policy_fingerprint(p / "train/initial-model.zip") for p in (features, zeros)]
    if fingerprints[0] != fingerprints[1]:
        raise ValueError("Initial policy tensors or network capacity differ")
    for path in (features, zeros):
        runtime = read(path / "train/runtime.json")
        if not runtime["static_area_filter"] or runtime["traffic_conflict_filter"]:
            raise ValueError("Training used the wrong effective action filters")
        for stage in ("initial", "100k", "300k", "final"):
            runtime = read(path / "evaluations" / stage / "summary.json")["runtime"]
            if not runtime["static_area_filter"] or runtime["traffic_conflict_filter"]:
                raise ValueError("Evaluation used the wrong effective action filters")
    return fingerprints[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "runs/cluster-ppo-static-v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh result file")
    plan_path = ROOT / "jobs/ppo_static_v1.json"
    plan = read(plan_path)
    cohorts = {}
    for arm in ("features", "zeros"):
        trials = []
        for seed in plan["seeds"]:
            folder = args.results / f"seed-{seed}" / arm
            trials.append({"seed": seed, "training": folder / "train",
                "classical": folder / "evaluations/classical",
                "evaluations": {stage: folder / "evaluations" / stage for stage in plan["evaluation"]["stages"]}})
        arm_plan = {**plan, "config": {**plan["config"], **plan["arms"][arm]}}
        cohorts[arm] = analyze(arm_plan, trials, ROOT, ROOT)
    if cohorts["features"]["training_environment"] != cohorts["zeros"]["training_environment"]:
        raise ValueError("The paired arms used different dependencies")
    if cohorts["features"]["classical_means"] != cohorts["zeros"]["classical_means"]:
        raise ValueError("The fixed classical benchmark changed across arms")
    worlds = plan["evaluation"]["worlds"]
    indices = np.random.default_rng(701).integers(0, worlds, size=(10000, worlds))
    pairs = []
    for seed in plan["seeds"]:
        folders = {arm: args.results / f"seed-{seed}" / arm for arm in ("features", "zeros")}
        identity = matched_pair(folders["features"], folders["zeros"])
        data = {arm: {stage: load_evaluation(folder / "evaluations" / stage)
                      for stage in ("initial", "final")} for arm, folder in folders.items()}
        reference = data["features"]["initial"]
        for arm in data:
            for loaded in data[arm].values():
                if loaded[2] != reference[2]:
                    raise ValueError("Feature-effect scenarios are not paired")
        effects = {}
        for metric in reference[3]:
            if not np.array_equal(reference[3][metric], data["zeros"]["initial"][3][metric]):
                raise ValueError("Matched neutral initial policies differ in observed behavior")
            gains = {arm: data[arm]["final"][3][metric].astype(float) -
                          data[arm]["initial"][3][metric].astype(float) for arm in data}
            effects[metric] = paired_effect(gains["features"], gains["zeros"], indices)
        pairs.append({"seed": seed, "initial_policy_fingerprint": identity,
                      "learning_effects": effects})
    result = {"experiment": plan["experiment"], "registered_protocol_sha256": digest(plan_path),
              "all_six_registered_runs_included": True, "cohorts": cohorts,
              "paired_feature_contribution": pairs,
              "features_minus_zeros_learning": aggregate_effects(pairs, plan["seeds"]),
              "interpretation": plan["interpretation"], "unseen_scenarios_used": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"results": str(args.out.resolve()),
                     "features_minus_zeros_learning": result["features_minus_zeros_learning"]}))


if __name__ == "__main__":
    main()
