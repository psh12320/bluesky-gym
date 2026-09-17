"""Policy/environment integration used by the original competition harness."""

import hashlib
import json
from pathlib import Path

import numpy as np

from atc.envs import make_env as build_environment
from atc.recipes import RECIPES

_CONFIGURATIONS = {}
_MODELS = {}


def _validate_configuration(kind, configuration):
    if kind not in {"sa", "ma"} or configuration.get("env") != kind:
        raise ValueError("Checkpoint track does not match the requested environment")
    if configuration.get("algorithm") not in {"sac", "ppo"}:
        raise ValueError("Unsupported checkpoint algorithm")
    recipe = configuration.get("recipe")
    if recipe not in RECIPES:
        raise ValueError("Unknown checkpoint recipe")
    settings = RECIPES[recipe]
    for key, value in settings.action_configuration().items():
        if configuration.get(key) != value:
            name = "decision-interval" if key == "decision_interval_seconds" else "speed-command"
            raise ValueError(f"Checkpoint {name} settings differ")
    if configuration.get("guard_static", False):
        if settings.goal_relative:
            raise ValueError("Static filtering does not support goal-relative control")
        if not settings.route_guided:
            from atc.projection import PROJECTION_REVISION, HORIZON_SECONDS, CLEARANCE_KM
            if (configuration.get("static_projection_revision") != PROJECTION_REVISION
                    or configuration.get("static_projection_horizon_seconds") != HORIZON_SECONDS
                    or configuration.get("static_projection_clearance_km") != CLEARANCE_KM):
                raise ValueError("Checkpoint static projection settings differ")
    if configuration.get("guard_traffic", False):
        from atc.traffic_projection import configuration as traffic_configuration
        if not configuration.get("guard_static") or settings.goal_relative or settings.route_guided:
            raise ValueError("Joint traffic filtering requires direct controls and static constraints")
        if any(configuration.get(key) != value for key, value in traffic_configuration().items()):
            raise ValueError("Checkpoint traffic projection settings differ")
    if settings.route_guided:
        from atc.routes import ROUTE_REVISION
        if configuration.get("route_revision", 1) != ROUTE_REVISION:
            raise ValueError("Checkpoint route revision differs; freeze a compatible candidate first")
    if settings.route_input:
        from atc.route_input import configuration as route_input_configuration
        if any(configuration.get(key) != value for key, value in route_input_configuration(settings.route_choice).items()):
            raise ValueError("Checkpoint route-input settings differ")
    if settings.route_residual:
        from atc.residual import configuration as residual_configuration
        if any(configuration.get(key) != value for key, value in residual_configuration(settings.fast_speed_reference).items()):
            raise ValueError("Checkpoint route-residual settings differ")
    if settings.conflict_features:
        from atc.conflicts import HORIZON_SECONDS
        if configuration.get("conflict_prediction_horizon_seconds") != HORIZON_SECONDS:
            raise ValueError("Checkpoint conflict prediction horizon differs")


def load_policy(kind, model_path=None):
    if model_path is None:
        _CONFIGURATIONS[kind] = {"recipe": "baseline", "guard_static": False}
        _MODELS.pop(kind, None)
        return lambda observation: np.zeros(2, dtype=np.float32)
    path = Path(model_path).resolve()
    if not path.is_file() and Path(str(path) + ".zip").is_file():
        path = Path(str(path) + ".zip")
    if not path.is_file():
        raise FileNotFoundError(path)
    for parent in (path.parent, path.parent.parent):
        config_path = parent / "config.json"
        if config_path.is_file():
            configuration = json.loads(config_path.read_text(encoding="utf-8"))
            break
    else:
        raise ValueError("Keep the original config.json alongside the checkpoint")
    _validate_configuration(kind, configuration)
    import torch
    from stable_baselines3 import PPO, SAC
    torch.set_num_threads(1)
    algorithm = configuration["algorithm"]
    kwargs = {"buffer_size": 1} if algorithm == "sac" else {}
    model = {"sac": SAC, "ppo": PPO}[algorithm].load(path, device="cpu", **kwargs)
    _CONFIGURATIONS[kind] = configuration
    _MODELS[kind] = model
    print(json.dumps({"submission_model": str(path), "env": kind, "algorithm": algorithm,
                      "recipe": configuration["recipe"],
                      "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
    return lambda observation: model.predict(observation, deterministic=True)[0]


def make_env(kind, n_agents=10):
    if n_agents != 10:
        raise ValueError("The competition evaluation requires 10 multi-agent aircraft")
    configuration = _CONFIGURATIONS.get(kind, {"recipe": "baseline", "guard_static": False})
    env = build_environment(kind, configuration["recipe"], bool(configuration.get("guard_static", False)),
                            bool(configuration.get("guard_traffic", False)))
    if kind in _MODELS:
        from stable_baselines3.common.utils import check_for_correct_spaces
        model = _MODELS[kind]
        if kind == "sa":
            observation_space, action_space = env.observation_space, env.action_space
        else:
            agent = env.possible_agents[0]
            observation_space, action_space = env.observation_space(agent), env.action_space(agent)
        try:
            check_for_correct_spaces(_SpaceView(observation_space, action_space),
                                     model.observation_space, model.action_space)
        except ValueError:
            env.close()
            raise
    return env


class _SpaceView:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
