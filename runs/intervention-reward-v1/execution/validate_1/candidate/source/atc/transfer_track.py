"""Evaluate an MA-trained SAC policy in SA with nine observed traffic slots."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from stable_baselines3 import SAC

from atc.envs import make_env
from atc.provenance import capture
from atc.recipes import RECIPES

TARGET_RECIPES = {"public_weights": "public_sa_transfer", "public_cpa": "public_cpa_sa_transfer",
                  "public_route_input": "public_route_input_sa_transfer",
                  "public_route_residual": "public_route_residual_sa_transfer",
                  "public_route_choice": "public_route_choice_sa_transfer",
                  "public_route_choice_residual": "public_route_choice_residual_sa_transfer",
                  "public_route_choice_speed20": "public_route_choice_speed20_sa_transfer",
                  "public_route_choice_fast_residual": "public_route_choice_fast_residual_sa_transfer",
                  "public_route_choice_speed20_residual": "public_route_choice_speed20_residual_sa_transfer",
                  "public_route_choice_interval5": "public_route_choice_interval5_sa_transfer",
                  "public_route_choice_fast_residual_interval5": "public_route_choice_fast_residual_interval5_sa_transfer",
                  "public_route_choice_fast_residual_reach250": "public_route_choice_fast_residual_reach250_sa_transfer"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_file() or args.out_dir.exists():
        parser.error("Use an existing source checkpoint and a new output directory")
    for parent in (args.model.parent, args.model.parent.parent):
        config_path = parent / "config.json"
        if config_path.is_file():
            configuration = json.loads(config_path.read_text(encoding="utf-8"))
            break
    else:
        parser.error("The original configuration was not found")
    if configuration.get("env") != "ma" or configuration.get("algorithm") != "sac":
        parser.error("Use a multi-agent SAC source checkpoint")
    from atc.submission import _validate_configuration
    _validate_configuration(configuration["env"], configuration)
    source_recipe = configuration.get("recipe")
    if source_recipe not in TARGET_RECIPES:
        parser.error("Unsupported transfer recipe")
    recipe = TARGET_RECIPES[source_recipe]
    torch.set_num_threads(1)
    model = SAC.load(args.model, device="cpu", buffer_size=1)
    env = make_env("sa", recipe, bool(configuration.get("guard_static", False)),
                   bool(configuration.get("guard_traffic", False)))
    try:
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            parser.error("The transferred policy's observation/action spaces differ")
        observation, _ = env.reset(seed=2026)
        world = env.unwrapped
        if (world.n_intruders != 10 or len(world.intruder_ids) != 10
                or len(world.scenario.intruder_routes) != 10 or world.intruder_obs.n != 9):
            raise AssertionError("SA must retain ten actual intruders and nine observation slots")
        action = model.predict(observation, deterministic=True)[0]
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise AssertionError("Transferred policy did not produce a valid action")
    finally:
        env.close()
    args.out_dir.mkdir(parents=True)
    capture(args.out_dir)
    shutil.copy2(args.model, args.out_dir / "model.zip")
    output = {"env": "sa", "algorithm": "sac", "recipe": recipe, "guard_static": bool(configuration.get("guard_static", False)),
              "gamma": model.gamma, "reward_kwargs": RECIPES[recipe].reward_kwargs(),
              "source_training_env": "ma", "source_model": str(args.model.resolve()),
              "source_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "source_timesteps": model.num_timesteps, "sa_training_transitions": 0,
              "initialization": "zero-shot MA-to-SA transfer; checkpoint bytes unchanged",
              "simulated_intruders": 10, "observed_traffic_slots": 9,
              "source_training_configuration": configuration}
    output["guard_traffic"] = bool(configuration.get("guard_traffic", False))
    output.update({key: configuration[key] for key in RECIPES[recipe].action_configuration()})
    if output["guard_traffic"]:
        from atc.traffic_projection import configuration as traffic_configuration
        output.update({key: configuration[key] for key in traffic_configuration()})
    if RECIPES[recipe].route_input:
        from atc.route_input import configuration as route_input_configuration
        output.update({key: configuration[key] for key in route_input_configuration(RECIPES[recipe].route_choice)})
    if RECIPES[recipe].route_residual:
        from atc.residual import configuration as residual_configuration
        output.update({key: configuration[key] for key in residual_configuration(RECIPES[recipe].fast_speed_reference)})
    if output["guard_static"]:
        for key in ("static_projection_revision", "static_projection_horizon_seconds", "static_projection_clearance_km"):
            output[key] = configuration[key]
    if RECIPES[recipe].conflict_features:
        output["conflict_prediction_horizon_seconds"] = configuration["conflict_prediction_horizon_seconds"]
    (args.out_dir / "config.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "source_training_configuration"}, indent=2))


if __name__ == "__main__":
    main()
