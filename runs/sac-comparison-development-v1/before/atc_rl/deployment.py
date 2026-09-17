"""Deploy shared PPO/MAPPO actors with recorded observation and action hooks."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo.utils import BaseParallelWrapper
from atc_rl.action_support import guard_settings, verify_guard_selection
from atc_rl.goal_offset import GoalOffsetActions, heading_commands
from atc_rl.world_worker import observation_recipe


def actor_space(original):
    if not isinstance(original, spaces.Box) or len(original.shape) != 1:
        raise ValueError("Expected a flattened local observation")
    return spaces.Box(np.append(original.low, 0).astype(np.float32),
                      np.append(original.high, 1).astype(np.float32), dtype=np.float32)


def timed_observation(observation, world, *, elapsed_seconds=None):
    elapsed = float(world.sim_time if elapsed_seconds is None else elapsed_seconds)
    if elapsed < 0 or not np.isfinite(elapsed):
        raise ValueError("Invalid simulator time")
    remaining = max(0.0, 1 - elapsed / world.episode_time_limit)
    return np.concatenate((np.asarray(observation, dtype=np.float32),
                           [remaining])).astype(np.float32)


class LocalInputsMA(BaseParallelWrapper):
    """Append the same finite-horizon clock used by the training collector."""
    def __init__(self, env):
        super().__init__(env)
        self._spaces = {a: actor_space(env.observation_space(a)) for a in env.possible_agents}

    def observation_space(self, agent):
        return self._spaces[agent]

    def _observe(self, observations):
        return {a: timed_observation(value, self.unwrapped) for a, value in observations.items()}

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._observe(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._observe(observations), rewards, terminations, truncations, infos


def dictionary_offsets(dictionary):
    if not isinstance(dictionary, spaces.Dict):
        raise ValueError("Expected the original dictionary observation layout")
    offsets = {}
    cursor = 0
    for name, space in dictionary.spaces.items():
        size = spaces.flatdim(space)
        offsets[name] = np.arange(cursor, cursor + size, dtype=np.int64)
        cursor += size
    return offsets, cursor


class LocalInputsSA(gym.ObservationWrapper):
    """Keep all ten intruders simulated, with the actor's trained input layout."""
    def __init__(self, env, mask_conflict_features=False):
        super().__init__(env)
        offsets, size = dictionary_offsets(env.unwrapped.observation_space)
        if env.observation_space.shape != (size,):
            raise ValueError("Flatten observations before deploying the actor")
        self.mask_indices = np.array([], dtype=np.int64)
        if mask_conflict_features:
            from atc.conflicts import FEATURE_NAMES
            if not set(FEATURE_NAMES).issubset(offsets):
                raise ValueError("Feature masking requires predictive observations")
            self.mask_indices = np.concatenate([offsets[name] for name in FEATURE_NAMES])
        self.observation_space = actor_space(env.observation_space)

    def observation(self, observation):
        result = np.asarray(observation).copy()
        result[self.mask_indices] = 0
        world = self.unwrapped
        return timed_observation(result, world,
                                 elapsed_seconds=world.metrics[world.agent]["flight_time"])


class GoalOffsetSA(gym.Wrapper):
    """Use the training heading representation in the original Gymnasium API."""
    def __init__(self, env):
        super().__init__(env)
        if env.unwrapped.d_heading != 45:
            raise ValueError("Checkpoint requires 45-degree heading command bounds")
        offsets, _ = dictionary_offsets(env.unwrapped.observation_space)
        self.cosine_index = int(offsets["cos_drift"][0])
        self.sine_index = int(offsets["sin_drift"][0])
        self._observation = None

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._observation = np.asarray(observation, dtype=np.float32).copy()
        return observation, info

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("Reset before taking a goal-offset action")
        command = heading_commands([action], [self._observation],
                                   self.cosine_index, self.sine_index)[0]
        result = self.env.step(command)
        self._observation = np.asarray(result[0], dtype=np.float32).copy()
        return result


class DecentralizedActor:
    """Accept one local observation; no joint critic state enters inference."""
    def __init__(self, model, configuration, record):
        self.model = model
        self.configuration = configuration
        self.record = record
        self.observation_space = model.observation_space["actor"]
        self.action_space = model.action_space
        self._critic_shape = model.observation_space["critic"].shape

    def __call__(self, observation):
        local = np.asarray(observation, dtype=np.float32)
        if local.shape != self.observation_space.shape or not np.isfinite(local).all():
            raise ValueError("Supply one finite local aircraft observation")
        inputs = {"actor": local, "critic": np.zeros(self._critic_shape, dtype=np.float32)}
        return self.model.predict(inputs, deterministic=True)[0]

    @classmethod
    def load(cls, model_path):
        import torch
        from stable_baselines3 import PPO
        from atc_rl.checkpoint_identity import verified_checkpoint
        path = Path(model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        configuration = json.loads((path.parent / "config.json").read_text(encoding="utf-8-sig"))
        if configuration["algorithm"] not in ("ppo", "mappo"):
            raise ValueError("Expected a shared PPO or MAPPO checkpoint")
        if configuration.get("mask_conflict_features", False) and not configuration.get("conflict_features", False):
            raise ValueError("Masked inputs require conflict features")
        if configuration.get("action_reference", "direct") not in ("direct", "goal_offset"):
            raise ValueError("Unsupported action representation")
        if configuration.get("decision_interval_seconds", 5) != 5:
            raise ValueError("Checkpoint action rate differs from the supported deployment")
        records = json.loads((path.parent / "checkpoints.json").read_text(encoding="utf-8-sig"))
        matches = [r for r in records if r["file"] == path.name]
        if len(matches) != 1 or verified_checkpoint(path.parent, matches[0]) != path:
            raise ValueError("Checkpoint integrity or identity differs")
        provenance = json.loads((path.parent / "provenance.json").read_text(encoding="utf-8-sig"))
        root = Path(__file__).resolve().parents[1]
        required_packages = ("atc", "core", "bluesky_gym", "bluesky_zoo")
        critical = ("atc_rl/policy.py", "atc_rl/goal_offset.py", "atc_rl/feature_mask.py", "atc_rl/traffic_scaling.py")
        for name, expected in provenance["source_sha256"].items():
            if name.split("/")[0] in required_packages or name in critical:
                file = root / name
                if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                    raise ValueError("Checkpoint environment/inference dependency changed: " + name)
        torch.set_num_threads(1)
        model = PPO.load(path, device="cpu")
        from atc_rl.exploration import validate_model
        validate_model(model, configuration)
        if model.policy.centralized != (configuration["algorithm"] == "mappo"):
            raise ValueError("Critic architecture differs from the recorded algorithm")
        return cls(model, configuration, matches[0])


def make_environment(kind, actor, n_agents=10):
    """Reconstruct permitted MDP hooks; keep the original scoring/scenario code."""
    if kind not in ("ma", "sa") or n_agents != 10:
        raise ValueError("Use the fixed single-agent or ten-agent competition configuration")
    from atc.envs import make_env
    from atc.heading_transport import attach_decimal_heading
    from atc.recipes import RECIPES
    config = actor.configuration
    from atc_rl.traffic_scaling import position_scale, ScaleTrafficMA, ScaleTrafficSA
    traffic_scale = position_scale(config)
    recipe = observation_recipe(config.get("guidance", False), config.get("conflict_features", False))
    if kind == "sa":
        recipe = replace(recipe, observation_traffic_slots=9)
    name = "onpolicy_deployment_" + kind
    RECIPES[name] = recipe
    guard_static, guard_traffic = guard_settings(config)
    env = attach_decimal_heading(make_env(kind, name, guard_static, guard_traffic))
    try:
        if traffic_scale != 1.0:
            env = ScaleTrafficMA(env, traffic_scale) if kind == "ma" else ScaleTrafficSA(env, traffic_scale)
        if kind == "ma":
            if config.get("mask_conflict_features", False):
                from atc_rl.feature_mask import MaskConflictFeatures
                env = MaskConflictFeatures(env)
            if config.get("action_reference", "direct") == "goal_offset":
                env = GoalOffsetActions(env)
            env = LocalInputsMA(env)
            observed, actions = env.observation_space(env.possible_agents[0]), env.action_space(env.possible_agents[0])
        else:
            env = LocalInputsSA(env, config.get("mask_conflict_features", False))
            if config.get("action_reference", "direct") == "goal_offset":
                env = GoalOffsetSA(env)
            observed, actions = env.observation_space, env.action_space
        if observed != actor.observation_space or actions != actor.action_space:
            raise ValueError("Native environment spaces differ from the checkpoint")
        world = env.unwrapped
        expected = {"episode_time_limit": 3000, "intrusion_distance": 5,
                    "distance_margin": 5, "action_frequency": 5, "sim_dt": 1, "n_obstacles": 5}
        if any(getattr(world, key) != value for key, value in expected.items()):
            raise ValueError("Fixed competition settings or trained action rate changed")
        if kind == "ma" and len(world.possible_agents) != 10:
            raise ValueError("Multi-agent population changed")
        if kind == "sa" and (world.n_intruders != 10 or world.intruder_obs.n != 9):
            raise ValueError("Single-agent transfer changed the world population or observed slots")
        env.deployment_runtime = {**verify_guard_selection(env, config),
            "traffic_position_scale": traffic_scale,
            "actor_uses_joint_context": False, "training_track": "ma", "evaluation_track": kind,
            "single_agent_transfer": kind == "sa", "observed_traffic_slots": world.intruder_obs.n,
            "simulated_scripted_intruders": world.n_intruders if kind == "sa" else 0, **expected}
        return env
    except BaseException:
        env.close()
        raise


_ACTORS = {}


def load_policy(kind, model_path=None):
    """Competition load_policy hook for explicitly selected learned checkpoints."""
    if kind not in ("ma", "sa") or model_path is None:
        raise ValueError("Select a track and a saved shared PPO/MAPPO checkpoint")
    actor = DecentralizedActor.load(model_path)
    _ACTORS[kind] = actor
    return actor


def make_env(kind, n_agents=10):
    """Competition make_env hook; call load_policy before constructing the world."""
    if kind not in _ACTORS:
        raise RuntimeError("Load the checkpoint before constructing its environment")
    return make_environment(kind, _ACTORS[kind], n_agents)
