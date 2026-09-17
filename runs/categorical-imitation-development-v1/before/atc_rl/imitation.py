"""Fit a local aircraft actor to recorded benchmark commands before any RL updates."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from atc_rl.demonstrations import load_demonstrations, require_disjoint, sha256


class ObservationOnlyEnvironment(gym.Env):
    """Supply policy spaces; supervised fitting must never step a simulator."""

    def __init__(self, actor_low, actor_high):
        low = np.asarray(actor_low, dtype=np.float32)
        high = np.asarray(actor_high, dtype=np.float32)
        if low.ndim != 1 or low.shape != high.shape or len(low) < 2:
            raise ValueError("Expected local observation bounds")
        self.observation_space = spaces.Dict({
            "actor": spaces.Box(low, high, dtype=np.float32),
            "critic": spaces.Box(
                np.concatenate((np.tile(low, 10), np.zeros(20, dtype=np.float32))),
                np.concatenate((np.tile(high, 10), np.ones(20, dtype=np.float32))),
                dtype=np.float32),
        })
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

    def reset(self, *args, **kwargs):
        raise RuntimeError("Observation-only environment cannot produce RL experience")

    def step(self, action):
        raise RuntimeError("Observation-only environment cannot produce RL experience")


def actor_parameters(policy):
    return [parameter for name, parameter in policy.named_parameters()
            if name.startswith(("mlp_extractor.policy_net.", "action_net."))]


def actor_mean(policy, local_observation):
    """AircraftPolicy has an identity local extractor; bypass unused critic storage."""
    from atc_rl.policy import AircraftPolicy
    if not isinstance(policy, AircraftPolicy) or policy.centralized:
        raise ValueError("Supervised pilot requires the standard local AircraftPolicy")
    return policy.action_net(policy.mlp_extractor.policy_net(local_observation))


def intervention_weights(intervened, fraction):
    flags = np.asarray(intervened)
    if flags.ndim != 1 or flags.dtype != np.bool_ or not len(flags):
        raise ValueError("Expected nonempty intervention flags")
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Intervention objective fraction must be in (0, 1)")
    proportion = float(flags.mean())
    if proportion in (0., 1.):
        return np.ones(len(flags), dtype=np.float32)
    return np.where(flags, fraction / proportion, (1. - fraction) / (1. - proportion)).astype(np.float32)


def fit_epoch(policy, observations, targets, weights, optimizer, permutation, batch_size):
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive batch size required")
    n = len(observations)
    if targets.shape != (n, 2) or weights.shape != (n,):
        raise ValueError("Supervised observations, actions and weights are not aligned")
    if len(permutation) != n or not np.array_equal(np.sort(permutation), np.arange(n)):
        raise ValueError("Every demonstration must occur exactly once per epoch")
    policy.set_training_mode(True)
    total, steps = 0., 0
    for start in range(0, n, batch_size):
        index = torch.as_tensor(permutation[start:start + batch_size], device=observations.device)
        predictions = actor_mean(policy, observations[index])
        squared_error = (predictions - targets[index]).square().mean(dim=1)
        loss = (squared_error * weights[index]).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite imitation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_parameters(policy), 1.)
        optimizer.step()
        total += float(loss.detach()) * len(index)
        steps += 1
    return total / n, steps


def action_errors(policy, data, batch_size=4096):
    arrays = data["arrays"]
    device = policy.device
    result = []
    policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(arrays["actor"]), batch_size):
            local = torch.as_tensor(arrays["actor"][start:start + batch_size], device=device)
            result.append(actor_mean(policy, local).clamp(-1, 1).cpu().numpy())
    predicted = np.concatenate(result)
    errors = predicted - arrays["teacher_action"]
    mse = np.mean(errors.astype(np.float64) ** 2, axis=1)
    intervention = arrays["intervened"]
    return {
        "action_mse": float(mse.mean()),
        "mean_absolute_heading_command_error_degrees": float(np.abs(errors[:, 0]).mean() * 45),
        "mean_absolute_speed_action_error": float(np.abs(errors[:, 1]).mean()),
        "intervention_action_mse": float(mse[intervention].mean()) if intervention.any() else None,
        "nominal_action_mse": float(mse[~intervention].mean()) if (~intervention).any() else None,
        "equal_world_mean_action_mse": float(np.mean([
            mse[data["world_index"] == i].mean() for i in np.unique(data["world_index"])])),
        "rows": len(mse), "worlds": len(data["scenarios"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=61400)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--intervention-fraction", type=float, default=.5)
    parser.add_argument("--initial-action-std", type=float, default=.05)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.seed < 0:
        parser.error("Positive budgets and nonnegative seed required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Positive finite learning rate required")
    if not math.isfinite(args.initial_action_std) or not 0 < args.initial_action_std <= 1:
        parser.error("Initial action std must be in (0,1]")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    torch.set_num_threads(1)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    training = load_demonstrations(args.training_data, "train")
    validation = load_demonstrations(args.validation_data, "validation")
    require_disjoint(training, validation)
    weights = intervention_weights(training["arrays"]["intervened"], args.intervention_fraction)
    if training["protocol"]["source_sha256"] != validation["protocol"]["source_sha256"]:
        raise ValueError("Training and validation demonstrations use different source")
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    from stable_baselines3 import PPO
    from atc_rl.policy import AircraftPolicy
    from atc_rl.initialization import neutral_action_mean
    from atc_rl.checkpoint_identity import policy_fingerprint
    from datetime import datetime, timezone
    root = Path(__file__).resolve().parents[1]
    source_hashes = {p.relative_to(root).as_posix(): sha256(p)
                    for package in ("atc", "atc_rl", "core", "bluesky_gym", "bluesky_zoo")
                    for p in (root / package).rglob("*.py")}
    teacher_modules = {"atc_rl/demonstrations.py", "atc_rl/world_worker.py", "atc_rl/action_support.py"}
    for key, value in training["protocol"]["source_sha256"].items():
        teacher_dependency = key.split("/")[0] in {"atc", "core", "bluesky_gym", "bluesky_zoo"} or key in teacher_modules
        if teacher_dependency and source_hashes.get(key) != value:
            raise ValueError("Teacher/observation implementation differs: " + key)
    config = {
        "algorithm": "behavior_cloning", "policy_family": "AircraftPolicy",
        "training_method": "behavior_cloning_only", "reinforcement_learning_performed": False,
        "seed": args.seed, "device": args.device, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "intervention_fraction": args.intervention_fraction,
        "guidance": True, "filter": False, "static_filter": True,
        "conflict_features": True, "mask_conflict_features": False,
        "traffic_position_scale": 1., "action_reference": "direct",
        "exploration": "gaussian", "initial_action_std": args.initial_action_std,
        "neutral_action_mean": False, "pretraining_initialization": "neutral mean",
        "reward_scale": .01, "progress_scale": 0.,
        "actor_widths": [128, 128], "critic_widths": [256, 256],
        "gamma": .996508469331006, "gae_lambda": .95,
        "teacher_has_combined_filter": True, "student_has_traffic_filter": False,
        "teacher_uses_privileged_state": True, "student_uses_local_inputs_only": True,
        "demonstration_transitions": len(training["arrays"]["actor"]),
        "validation_transitions": len(validation["arrays"]["actor"]),
        "dataset_provenance": {name: {
            "directory": data["directory"], "seed": data["protocol"]["seed"],
            "complete_sha256": sha256(Path(data["directory"]) / "complete.json"),
            "protocol_sha256": data["completion"]["protocol_sha256"],
            "world_manifest_sha256": data["completion"]["world_manifest_sha256"],
        } for name, data in (("train", training), ("validation", validation))},
        "checkpoint_selection": "Fixed final epoch; validation is reported but does not select epochs.",
        "budget_interpretation": "live_transitions and PPO optimizer_steps are zero; supervised updates and repeated examples are recorded separately.",
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (directory / "provenance.json").write_text(json.dumps({"source_sha256": source_hashes}, indent=2), encoding="utf-8")
    environment = ObservationOnlyEnvironment(training["actor_low"], training["actor_high"])
    model = PPO(AircraftPolicy, environment, seed=args.seed, device=args.device,
                n_steps=2, batch_size=2, n_epochs=1, gamma=config["gamma"], gae_lambda=.95,
                policy_kwargs={"centralized": False, "log_std_init": math.log(args.initial_action_std)})
    neutral_action_mean(model)
    state_before = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    optimizer = torch.optim.Adam(actor_parameters(model.policy), lr=args.learning_rate)
    x = torch.as_tensor(training["arrays"]["actor"], device=model.device)
    y = torch.as_tensor(training["arrays"]["teacher_action"], device=model.device)
    w = torch.as_tensor(weights, device=model.device)
    rng = np.random.default_rng(args.seed)
    updates = examples = 0
    checkpoints = []

    def save(name, epoch):
        path = directory / name
        model.save(path)
        checkpoints.append({
            "file": name, "sha256": sha256(path), "policy_fingerprint": policy_fingerprint(path),
            "live_transitions": 0, "counted_transitions": 0, "padded_transitions": 0,
            "optimizer_steps": 0, "policy_epochs": 0, "rollouts": 0,
            "training_method": "behavior_cloning", "supervised_epochs": epoch,
            "supervised_optimizer_steps": updates, "supervised_examples_seen": examples,
            "demonstration_transitions": config["demonstration_transitions"],
        })
        (directory / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2), encoding="utf-8")

    save("untrained-model.zip", 0)
    curve = [{"epoch": 0, "supervised_optimizer_steps": 0,
              "train": action_errors(model.policy, training), "validation": action_errors(model.policy, validation)}]
    with (directory / "learning.jsonl").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(curve[0]) + "\n")
        for epoch in range(1, args.epochs + 1):
            loss, steps = fit_epoch(model.policy, x, y, w, optimizer, rng.permutation(len(x)), args.batch_size)
            updates += steps
            examples += len(x)
            row = {"epoch": epoch, "supervised_optimizer_steps": updates,
                   "supervised_examples_seen": examples, "weighted_training_loss": loss,
                   "train": action_errors(model.policy, training), "validation": action_errors(model.policy, validation)}
            curve.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps(row), flush=True)
    changed = [k for k, value in model.policy.state_dict().items() if not torch.equal(value, state_before[k])]
    if not changed or any(not key.startswith(("mlp_extractor.policy_net.", "action_net.")) for key in changed):
        raise ValueError("Imitation must change only actor mean parameters")
    if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
        raise ValueError("Supervised fitting unexpectedly changed RL accounting or optimizer")
    save("model.zip", args.epochs)
    model.policy.set_training_mode(False)
    restored = PPO.load(directory / "model.zip", device=model.device)
    probe = {"actor": training["arrays"]["actor"][:32],
             "critic": np.zeros((min(32, len(x)), environment.observation_space["critic"].shape[0]), dtype=np.float32)}
    if not np.array_equal(model.predict(probe, deterministic=True)[0], restored.predict(probe, deterministic=True)[0]):
        raise ValueError("Serialized imitation policy changed deterministic actions")
    for name, value in source_hashes.items():
        if sha256(root / name) != value:
            raise ValueError("Source changed during imitation fitting")
    summary = {
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_method": "behavior_cloning_only", "reinforcement_learning_updates": 0,
        "supervised_optimizer_steps": updates, "supervised_examples_seen": examples,
        "demonstration_transitions": len(x), "validation_transitions": len(validation["arrays"]["actor"]),
        "epochs": args.epochs, "changed_parameter_names": changed,
        "actor_mean_only_changed": True, "critic_and_noise_unchanged": True,
        "initial_errors": curve[0], "final_errors": curve[-1],
        "serialization_actions_identical": True, "model_sha256": checkpoints[-1]["sha256"],
        "performance_claim": False,
        "next_validation": "Evaluate the saved learned policy in the simulator, then measure PPO gains against its own pretrained starting policy.",
    }
    (directory / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
