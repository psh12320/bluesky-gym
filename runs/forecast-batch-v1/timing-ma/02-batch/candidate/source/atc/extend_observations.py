"""Extend a SAC checkpoint with conflict features while preserving its initial policy."""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import FlattenExtractor

from atc.conflicts import HORIZON_SECONDS
from atc.encoders import PredictionResidual
from atc.envs import make_env
from atc.provenance import capture
from atc.recipes import RECIPES

TARGET_RECIPES = {"goal_relative": "goal_relative_cpa", "public_weights": "public_cpa"}


class SpaceOnlyEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), {}

    def step(self, action):
        raise RuntimeError("This environment supplies spaces only; use BlueSky for training")


def feature_indices(original_space, extended_space):
    """Find original feature coordinates in the augmented flattened Dict."""
    offsets, offset = {}, 0
    for name, space in extended_space.spaces.items():
        offsets[name] = list(range(offset, offset + spaces.flatdim(space)))
        offset += spaces.flatdim(space)
    original = []
    for name, space in original_space.spaces.items():
        if name not in extended_space.spaces or space != extended_space[name]:
            raise ValueError(f"Original observation field changed: {name}")
        original.extend(offsets[name])
    used = set(original)
    added = [index for index in range(offset) if index not in used]
    if not added:
        raise ValueError("The target observation has no additional features")
    return original, added


def extend_model(source, target_space, original_indices, added_indices, *, enabled=True, buffer_size=250000):
    if not isinstance(source.actor.features_extractor, FlattenExtractor):
        raise ValueError("Only a plain flattened-observation SAC source is supported")
    if len(original_indices) != source.observation_space.shape[0]:
        raise ValueError("Source observation size does not match the retained feature coordinates")
    kwargs = dict(source.policy_kwargs)
    kwargs.update(features_extractor_class=PredictionResidual,
                  features_extractor_kwargs={"original_indices": original_indices,
                                            "added_indices": added_indices, "enabled": enabled})
    model = SAC("MlpPolicy", SpaceOnlyEnv(target_space, source.action_space),
                learning_rate=float(source.actor.optimizer.param_groups[0]["lr"]),
                gamma=source.gamma, tau=source.tau, batch_size=source.batch_size,
                ent_coef=source.ent_coef, target_entropy=source.target_entropy,
                learning_starts=source.learning_starts, train_freq=1,
                gradient_steps=source.gradient_steps, buffer_size=1,
                replay_buffer_class=source.replay_buffer_class,
                policy_kwargs=kwargs, seed=source.seed, device="cpu", verbose=0)
    missing, unexpected = model.policy.load_state_dict(source.policy.state_dict(), strict=False)
    if unexpected or any(".features_extractor." not in key for key in missing):
        raise ValueError(f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        if source.log_ent_coef is not None:
            model.log_ent_coef.copy_(source.log_ent_coef)
        elif source.ent_coef_tensor is not None:
            model.ent_coef_tensor.copy_(source.ent_coef_tensor)
    model.num_timesteps = source.num_timesteps
    model._n_updates = source._n_updates
    # Replay is deliberately absent. Save the capacity to allocate on the next
    # training load, without allocating a large unused buffer during conversion.
    model.buffer_size = buffer_size
    return model


def observation_dictionary(kind, recipe):
    env = make_env(kind, recipe)
    try:
        if kind == "sa":
            return env.env.observation_space
        return env.env.observation_space(env.possible_agents[0])
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--freeze-predictions", action="store_true", help="Keep the added projection at zero for the matched control")
    parser.add_argument("--buffer-size", type=int)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("Source checkpoint does not exist")
    if args.out_dir.exists():
        parser.error("Output directory already exists")
    for parent in (args.model.parent, args.model.parent.parent):
        path = parent / "config.json"
        if path.is_file():
            configuration = json.loads(path.read_text(encoding="utf-8"))
            break
    else:
        parser.error("Source checkpoint needs its original config.json")
    from atc.submission import _validate_configuration
    _validate_configuration(configuration["env"], configuration)
    recipe = configuration.get("recipe")
    if configuration.get("algorithm") != "sac" or recipe not in TARGET_RECIPES:
        parser.error("Use a SAC checkpoint from goal_relative or public_weights")
    kind, target_recipe = configuration["env"], TARGET_RECIPES[recipe]
    original_space = observation_dictionary(kind, recipe)
    target_dict = observation_dictionary(kind, target_recipe)
    original_indices, added_indices = feature_indices(original_space, target_dict)
    torch.set_num_threads(1)
    source = SAC.load(args.model, device="cpu", buffer_size=1)
    if source.gamma != RECIPES[target_recipe].gamma:
        parser.error("Checkpoint discount differs from the target recipe")
    if configuration.get("reward_kwargs", RECIPES[recipe].reward_kwargs()) != RECIPES[target_recipe].reward_kwargs():
        parser.error("Checkpoint reward weights differ from the target recipe")
    if source.observation_space != spaces.flatten_space(original_space):
        parser.error("Checkpoint does not match its declared original observation space")
    capacity = args.buffer_size if args.buffer_size is not None else (configuration.get("actual_buffer_size") or configuration.get("buffer_size", 250000))
    if capacity < 1:
        parser.error("Replay capacity must be positive")
    enabled = not args.freeze_predictions
    model = extend_model(source, spaces.flatten_space(target_dict), original_indices, added_indices,
                         enabled=enabled, buffer_size=capacity)
    args.out_dir.mkdir(parents=True)
    capture(args.out_dir)
    output_configuration = {"env": kind, "algorithm": "sac", "recipe": target_recipe,
                            "guard_static": bool(configuration.get("guard_static", False)), "gamma": source.gamma,
                            "reward_kwargs": RECIPES[target_recipe].reward_kwargs(),
                            "conflict_prediction_horizon_seconds": HORIZON_SECONDS,
                            "buffer_size": capacity, "actual_buffer_size": capacity,
                            "initialization": "zero prediction residual from an existing SAC checkpoint",
                            "source_model": str(args.model.resolve()),
                            "source_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                            "source_timesteps": source.num_timesteps,
                            "source_observation_shape": list(source.observation_space.shape),
                            "observation_shape": list(model.observation_space.shape),
                            "prediction_projection_trainable": enabled,
                            "optimizer_state": "reset", "replay_state": "empty",
                            "transitions_trained_after_extension": 0,
                            "source_entropy_coefficient": float(source.log_ent_coef.exp().item()) if source.log_ent_coef is not None else float(source.ent_coef_tensor.item())}
    output_configuration["guard_traffic"] = bool(configuration.get("guard_traffic", False))
    if output_configuration["guard_traffic"]:
        from atc.traffic_projection import configuration as traffic_configuration
        output_configuration.update({key: configuration[key] for key in traffic_configuration()})
    if output_configuration["guard_static"]:
        for key in ("static_projection_revision", "static_projection_horizon_seconds", "static_projection_clearance_km"):
            output_configuration[key] = configuration[key]
    model.save(args.out_dir / "model")
    (args.out_dir / "config.json").write_text(json.dumps(output_configuration, indent=2), encoding="utf-8")
    print(json.dumps(output_configuration, indent=2))
    model.env.close()


if __name__ == "__main__":
    main()
