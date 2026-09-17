"""Audit fresh SAC model identity, sources, training counts and optimizer activity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new audit output path")
    directory = args.run.resolve()
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8-sig"))
    config, summary, manifest = read("config.json"), read("training_summary.json"), read("checkpoints.json")
    provenance = read("provenance.json")
    if config["algorithm"] != "sac" or summary["algorithm"] != "sac":
        raise ValueError("Expected fresh SAC artifacts")
    import torch
    from atc_rl.sac_support import load_sac_for_inference
    from atc_rl.checkpoint_identity import verified_checkpoint
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(directory / "source.zip") as archive:
        for name, digest in provenance["source_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Archived source mismatch: " + name)
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Current audit source differs: " + name)
    for record in manifest:
        verified_checkpoint(directory, record)
        if record["counted_transitions"] != record["live_transitions"] + record["padded_transitions"]:
            raise ValueError("Checkpoint transition counts disagree")
        counts = [record[key] for key in ("optimizer_steps", "actor_optimizer_steps", "entropy_optimizer_steps", "gradient_rounds")]
        if len(set(counts)) != 1:
            raise ValueError("Checkpoint optimizer counts disagree")
    initial_records = [r for r in manifest if r["file"] == "initial-model.zip"]
    final_records = [r for r in manifest if r["file"] == "model.zip"]
    if len(initial_records) != 1 or len(final_records) != 1:
        raise ValueError("Expected unique initial and final models")
    initial_record, final_record = initial_records[0], final_records[0]
    if initial_record["live_transitions"] or initial_record["counted_transitions"] or initial_record["optimizer_steps"]:
        raise ValueError("Initial model was already trained")
    if final_record["optimizer_steps"] <= 0 or final_record["live_transitions"] <= 0:
        raise ValueError("No learned SAC policy recorded")
    if final_record["sha256"] != summary["model_sha256"]:
        raise ValueError("Summary model identity differs")
    initial = load_sac_for_inference(directory / "initial-model.zip", config)
    final = load_sac_for_inference(directory / "model.zip", config)
    if initial.num_timesteps or initial._n_updates:
        raise ValueError("Initial serialized model already trained")
    if final.num_timesteps != final_record["counted_transitions"] or final._n_updates != final_record["gradient_rounds"]:
        raise ValueError("Serialized training counters differ")
    for prefix in ("actor", "critic"):
        first, last = getattr(initial, prefix).state_dict(), getattr(final, prefix).state_dict()
        if not any(not torch.equal(first[k], last[k]) for k in first):
            raise ValueError(prefix + " parameters did not change")
    if any(not torch.isfinite(p).all() for p in final.policy.parameters()):
        raise ValueError("Nonfinite final parameters")
    if torch.count_nonzero(initial.actor.mu.weight) or torch.count_nonzero(initial.actor.mu.bias):
        raise ValueError("Initial deterministic actor is not neutral")
    with (directory / "learning.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Missing learning records")
    for key in ("live_transitions", "padded_transitions", "counted_transitions", "optimizer_steps",
                "actor_optimizer_steps", "entropy_optimizer_steps", "gradient_rounds"):
        if int(rows[-1][key]) != summary[key] or summary[key] != final_record[key]:
            raise ValueError("Learning, summary and checkpoint counters disagree: " + key)
        sequence = [int(r[key]) for r in rows]
        if sequence != sorted(sequence):
            raise ValueError("Nonmonotonic learning counters: " + key)
    with (directory / "training-aircraft.csv").open(newline="", encoding="utf-8") as stream:
        completed = sum(1 for _ in csv.DictReader(stream))
    with (directory / "training-returns.csv").open(newline="", encoding="utf-8") as stream:
        returns = sum(1 for _ in csv.DictReader(stream))
    if completed != returns or completed != summary["aircraft_completed"]:
        raise ValueError("Completed-aircraft records disagree")
    if summary["status"] == "complete" and summary["live_transitions"] < config["live_steps"]:
        raise ValueError("Completed run did not reach its requested budget")
    result = {"algorithm": "sac", "status": summary["status"], "model_sha256": final_record["sha256"],
              "live_transitions": summary["live_transitions"], "optimizer_steps": summary["optimizer_steps"],
              "verified_checkpoints": len(manifest), "verified_source_files": len(provenance["source_sha256"]),
              "learning_rows": len(rows), "aircraft_completed": completed,
              "actor_parameter_count": sum(p.numel() for p in final.actor.parameters()),
              "critic_parameter_count": sum(p.numel() for p in final.critic.parameters()),
              "actor_and_critic_changed": True, "actor_and_critics_use_local_inputs_only": True,
              "finite_task_deadline": True, "historical_replay_loaded": False, "policy_quality_assessed": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
