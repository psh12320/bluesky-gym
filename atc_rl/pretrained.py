"""Initialize PPO or MAPPO from a verified supervised actor, retaining its learning history."""
from pathlib import Path
import hashlib
import json
import random
import shutil

import numpy as np
import torch
from stable_baselines3 import PPO
from atc_rl.checkpoint_identity import policy_fingerprint, verified_checkpoint
from atc_rl.demonstrations import sha256
from atc_rl.actor_reference import is_actor_parameter

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
    if isinstance(parent, spaces.MultiDiscrete) or isinstance(target, spaces.MultiDiscrete):
        expected=spaces.MultiDiscrete([20,3])
        if parent!=expected or target!=expected:
            raise ValueError("Expected identical registered heading/speed categorical spaces")
        return {"kind":"categorical", "nvec":[20,3], "start":[0,0], "parent_dtype":str(parent.dtype), "ppo_dtype":str(target.dtype)}
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


def critic_fingerprint(state):
    """Identify the critic tensors retained during actor-only initialization."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if is_actor_parameter(name):
            continue
        tensor = value.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        data = tensor.numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


def initialize_from_pretrained(model, checkpoint, config, run_directory):
    centralized = config["algorithm"] == "mappo"
    if config["algorithm"] not in ("ppo", "mappo") or model.policy.centralized != centralized:
        raise ValueError("Recorded PPO/MAPPO algorithm must match the target critic")
    if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
        raise ValueError("Pretraining requires a fresh PPO run")
    if config.get("neutral_action_mean") or config.get("actor_reference"):
        raise ValueError("Pretraining cannot be combined with another actor initialization")
    source_config, summary, record = inspect_parent(checkpoint)
    for key in MATCHING_SETTINGS:
        if source_config.get(key) != config.get(key):
            raise ValueError("Pretrained policy changes setting " + key)
    if config.get("exploration")=="categorical":
        from atc_rl.exploration import require_matching
        require_matching(source_config,config)
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
    selected = {name for name in state if not centralized or is_actor_parameter(name)}
    target_names = {name for name in expected if not centralized or is_actor_parameter(name)}
    if not selected or selected != target_names or any(state[k].shape != expected[k].shape for k in selected):
        raise ValueError("Pretrained parameter layout differs")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite pretrained parameters")
    critic_before = critic_fingerprint(expected) if centralized else None
    archive = Path(run_directory).resolve() / "pretraining"
    archive.mkdir(exist_ok=False)
    parent = Path(checkpoint).resolve().parent
    copies = {}
    for name in ("model.zip", "config.json", "training_summary.json", "checkpoints.json", "provenance.json"):
        shutil.copyfile(parent / name, archive / name)
        copies[name] = sha256(archive / name)
    if centralized:
        replacement = {name: state[name] if name in selected else value for name, value in expected.items()}
        model.policy.load_state_dict(replacement, strict=True)
        if critic_fingerprint(model.policy.state_dict()) != critic_before:
            raise ValueError("Actor transfer changed the fresh centralized critic")
    else:
        model.policy.load_state_dict(state, strict=True)
    if model.policy.optimizer.state or model.num_timesteps or model._n_updates:
        raise ValueError("PPO accounting must remain independent of supervised fitting")
    lineage = {
        "method": "behavior_cloning", "original_directory": str(parent),
        "archived_directory": "pretraining", "archived_files": copies, "parent_checkpoint": record,
        **{k: summary[k] for k in ("demonstration_transitions", "supervised_optimizer_steps", "supervised_examples_seen")},
        "initial_policy_is_untrained": False, "rl_live_transitions_at_initialization": 0,
        "ppo_optimizer_inherited": False,
        "action_space_compatibility": action_compatibility,
    }
    if centralized:
        lineage.update(transfer_scope="actor_only", actor_parameter_names=sorted(selected),
                       fresh_critic_state_sha256=critic_before,
                       centralized_critic_initialization="Fresh MAPPO critic tensors retained unchanged",
                       learner_rng="Preserved after fresh MAPPO construction; not asserted identical to PPO sampling")
    return lineage


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
    scope = lineage.get("transfer_scope", "full_policy")
    if scope not in ("full_policy", "actor_only"):
        raise ValueError("Unsupported supervised transfer scope")
    centralized = config.get("algorithm") == "mappo"
    if config.get("algorithm") not in ("ppo", "mappo") or (scope == "actor_only") != centralized:
        raise ValueError("Supervised transfer scope differs from PPO/MAPPO algorithm")
    if scope == "full_policy" and policy_fingerprint(directory / "initial-model.zip") != record["policy_fingerprint"]:
        raise ValueError("PPO starting tensors differ from the pretrained parent")
    parent = PPO.load(archive / "model.zip", device="cpu")
    initial = PPO.load(directory / "initial-model.zip", device="cpu")
    if parent.policy.centralized or initial.policy.centralized != centralized:
        raise ValueError("Archived supervised critic architecture differs")
    if initial.num_timesteps or initial._n_updates or initial.policy.optimizer.state:
        raise ValueError("Supervised initial checkpoint contains PPO training state")
    if scope == "actor_only":
        parent_state, initial_state = parent.policy.state_dict(), initial.policy.state_dict()
        actor_names = sorted(name for name in initial_state if is_actor_parameter(name))
        if actor_names != lineage.get("actor_parameter_names") or actor_names != sorted(name for name in parent_state if is_actor_parameter(name)):
            raise ValueError("Supervised actor parameter layout differs")
        if any(not torch.equal(initial_state[name], parent_state[name]) for name in actor_names):
            raise ValueError("MAPPO starting actor differs from the supervised parent")
        if critic_fingerprint(initial_state) != lineage.get("fresh_critic_state_sha256"):
            raise ValueError("MAPPO starting critic differs from its retained fresh tensors")
    if initial.observation_space != parent.observation_space:
        raise ValueError("Archived parent and PPO observation spaces differ")
    compatibility = action_space_compatibility(parent.action_space, initial.action_space)
    if compatibility != lineage.get("action_space_compatibility"):
        raise ValueError("Action-space compatibility record differs from saved models")
    return lineage
