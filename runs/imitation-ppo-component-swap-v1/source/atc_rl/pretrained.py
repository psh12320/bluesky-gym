"""Initialize fresh PPO from a verified supervised policy, retaining its learning history."""
from pathlib import Path
import json
import random
import shutil

import numpy as np
import torch
from stable_baselines3 import PPO
from atc_rl.checkpoint_identity import policy_fingerprint, verified_checkpoint
from atc_rl.demonstrations import sha256

READ = lambda p: json.loads(Path(p).read_text(encoding="utf-8-sig"))
MATCHING_SETTINGS = ("guidance", "filter", "static_filter", "conflict_features", "mask_conflict_features",
                     "traffic_position_scale", "action_reference", "exploration", "initial_action_std",
                     "actor_widths", "critic_widths")


def inspect_parent(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    parent = checkpoint.parent
    config, summary = READ(parent / "config.json"), READ(parent / "training_summary.json")
    matches = [r for r in READ(parent / "checkpoints.json") if r["file"] == checkpoint.name]
    if len(matches) != 1:
        raise ValueError("Expected one original checkpoint record")
    record = matches[0]
    verified_checkpoint(parent, record)
    if config.get("algorithm") != "behavior_cloning" or config.get("training_method") != "behavior_cloning_only":
        raise ValueError("Expected an explicitly identified behavior-cloning parent")
    if config.get("reinforcement_learning_performed") is not False or summary.get("reinforcement_learning_updates") != 0:
        raise ValueError("Supervised parent must not already contain RL training")
    if summary.get("status") != "complete" or not summary.get("critic_and_noise_unchanged") or not summary.get("serialization_actions_identical"):
        raise ValueError("Parent fitting or serialization checks are incomplete")
    if checkpoint.name != "model.zip" or summary["model_sha256"] != record["sha256"]:
        raise ValueError("Use the registered final supervised checkpoint")
    for key in ("supervised_optimizer_steps", "supervised_examples_seen", "demonstration_transitions"):
        if record[key] != summary[key] or record[key] <= 0:
            raise ValueError("Inconsistent supervised experience")
    if record["supervised_epochs"] != summary["epochs"] or summary["epochs"] != config["epochs"]:
        raise ValueError("Incomplete supervised budget")
    if any(record[k] != 0 for k in ("live_transitions", "counted_transitions", "optimizer_steps")):
        raise ValueError("Unexpected parent RL counters")
    if policy_fingerprint(checkpoint) != record["policy_fingerprint"]:
        raise ValueError("Parent policy tensors changed")
    return config, summary, record


def action_space_compatibility(parent, target):
    """Allow only equivalent normalized heading/speed Boxes in float32/float64."""
    from gymnasium import spaces
    for space in (parent, target):
        if (not isinstance(space, spaces.Box) or space.shape != (2,) or
                space.dtype not in (np.dtype("float32"), np.dtype("float64")) or
                not np.all(space.low == -1) or not np.all(space.high == 1)):
            raise ValueError("Expected normalized floating heading/speed action bounds")
    if not np.array_equal(parent.low, target.low) or not np.array_equal(parent.high, target.high):
        raise ValueError("Pretrained action bounds differ")
    return {"parent_dtype": str(parent.dtype), "ppo_dtype": str(target.dtype),
            "shape": [2], "low": [-1., -1.], "high": [1., 1.],
            "only_floating_storage_dtype_may_differ": True}


def initialize_from_pretrained(model, checkpoint, config, run_directory):
    if config["algorithm"] != "ppo" or model.policy.centralized:
        raise ValueError("Full supervised initialization currently supports local-critic PPO only")
    if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
        raise ValueError("Pretraining requires a fresh PPO run")
    if config.get("neutral_action_mean") or config.get("actor_reference"):
        raise ValueError("Pretraining cannot be combined with another actor initialization")
    source_config, summary, record = inspect_parent(checkpoint)
    for key in MATCHING_SETTINGS:
        if source_config.get(key) != config.get(key):
            raise ValueError("Pretrained policy changes setting " + key)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
            restored = PPO.load(checkpoint, device=model.device)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if restored.num_timesteps or restored._n_updates or restored.policy.optimizer.state:
        raise ValueError("Serialized parent contains unexpected PPO training state")
    if restored.observation_space != model.observation_space:
        raise ValueError("Pretrained observation spaces differ")
    action_compatibility = action_space_compatibility(restored.action_space, model.action_space)
    if restored.policy.centralized:
        raise ValueError("Parent must have a local critic")
    state, expected = restored.policy.state_dict(), model.policy.state_dict()
    if state.keys() != expected.keys() or any(state[k].shape != expected[k].shape for k in state):
        raise ValueError("Pretrained parameter layout differs")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite pretrained parameters")
    archive = Path(run_directory).resolve() / "pretraining"
    archive.mkdir(exist_ok=False)
    parent = Path(checkpoint).resolve().parent
    copies = {}
    for name in ("model.zip", "config.json", "training_summary.json", "checkpoints.json", "provenance.json"):
        shutil.copyfile(parent / name, archive / name)
        copies[name] = sha256(archive / name)
    model.policy.load_state_dict(state, strict=True)
    if model.policy.optimizer.state or model.num_timesteps or model._n_updates:
        raise ValueError("PPO accounting must remain independent of supervised fitting")
    return {
        "method": "behavior_cloning", "original_directory": str(parent),
        "archived_directory": "pretraining", "archived_files": copies, "parent_checkpoint": record,
        **{k: summary[k] for k in ("demonstration_transitions", "supervised_optimizer_steps", "supervised_examples_seen")},
        "initial_policy_is_untrained": False, "rl_live_transitions_at_initialization": 0,
        "ppo_optimizer_inherited": False,
        "action_space_compatibility": action_compatibility,
    }


def audit_pretraining(run_directory, config):
    lineage = config.get("pretraining")
    if lineage is None:
        if config.get("pretrained_model"):
            raise ValueError("Pretrained run is missing its learning history")
        return None
    directory = Path(run_directory).resolve()
    archive = (directory / lineage["archived_directory"]).resolve()
    if archive.parent != directory:
        raise ValueError("Unsafe pretraining archive path")
    for name, expected in lineage["archived_files"].items():
        path = (archive / name).resolve()
        if path.parent != archive or sha256(path) != expected:
            raise ValueError("Archived supervised lineage changed")
    _, summary, record = inspect_parent(archive / "model.zip")
    if record != lineage["parent_checkpoint"]:
        raise ValueError("Parent checkpoint differs from recorded lineage")
    for key in ("demonstration_transitions", "supervised_optimizer_steps", "supervised_examples_seen"):
        if summary[key] != lineage[key]:
            raise ValueError("Supervised budget differs from the parent")
    if lineage["initial_policy_is_untrained"] or lineage["rl_live_transitions_at_initialization"] != 0 or lineage["ppo_optimizer_inherited"]:
        raise ValueError("Invalid pretraining accounting")
    if policy_fingerprint(directory / "initial-model.zip") != record["policy_fingerprint"]:
        raise ValueError("PPO starting tensors differ from the pretrained parent")
    parent = PPO.load(archive / "model.zip", device="cpu")
    initial = PPO.load(directory / "initial-model.zip", device="cpu")
    if initial.observation_space != parent.observation_space:
        raise ValueError("Archived parent and PPO observation spaces differ")
    compatibility = action_space_compatibility(parent.action_space, initial.action_space)
    if compatibility != lineage.get("action_space_compatibility"):
        raise ValueError("Action-space compatibility record differs from saved models")
    return lineage
