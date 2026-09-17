from functools import partial
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper

from atc.recipes import RECIPES


class FlattenObservations(BaseParallelWrapper):
    """Expose each aircraft's dictionary observation as a fixed vector."""

    def __init__(self, env):
        super().__init__(env)
        self._spaces = {
            agent: spaces.flatten_space(env.observation_space(agent))
            for agent in env.possible_agents
        }

    def observation_space(self, agent):
        return self._spaces[agent]

    def _flatten(self, observations):
        return {
            agent: spaces.flatten(self.env.observation_space(agent), observation)
            for agent, observation in observations.items()
        }

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._flatten(observations), infos

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        return self._flatten(observations), rewards, terminated, truncated, infos


def make_env(kind, recipe="baseline", guard_static=False, guard_traffic=False):
    import bluesky as bs
    if bs.sim is None:
        runtime = Path(__file__).resolve().parents[1] / "runs" / "simulator"
        runtime.mkdir(parents=True, exist_ok=True)
        bs.init(mode="sim", detached=True, workdir=str(runtime))
    # Recipes change permitted MDP hooks; scenario and scoring defaults stay fixed.
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe: {recipe}")
    settings = RECIPES[recipe]
    guard_static = bool(guard_static or guard_traffic)
    if guard_traffic and (settings.goal_relative or settings.route_guided):
        raise ValueError("Joint traffic filtering requires direct heading/speed control")
    if guard_static and settings.goal_relative:
        raise ValueError("Static filtering does not support goal-relative control")
    kwargs = settings.reward_kwargs()
    kwargs.update(settings.action_kwargs())
    if settings.fast_speed_reference:
        kwargs["fast_speed_reference"] = True
    if kind == "sa":
        from bluesky_gym.envs.competition_env import CompetitionEnv
        base = CompetitionEnv
    elif kind == "ma":
        from bluesky_zoo.competition_v0 import CompetitionZooEnv
        base = CompetitionZooEnv
    else:
        raise ValueError(f"Unknown environment: {kind}")
    if settings.goal_relative:
        from atc.control import GoalRelativeControl
        class GoalRelativeEnv(GoalRelativeControl, base):
            pass
        base = GoalRelativeEnv
    if settings.route_guided:
        from atc.routes import RouteGuidance
        class RouteGuidedEnv(RouteGuidance, base):
            pass
        base = RouteGuidedEnv
        kwargs.update(route_sector_clearance=settings.route_sector_clearance, guard_static=guard_static)
    if guard_static and not settings.route_guided:
        from atc.projection import StaticActionProjection
        projection_class = StaticActionProjection
        if guard_traffic:
            from atc.traffic_projection import TrafficActionProjection
            projection_class = TrafficActionProjection
        class ProjectedEnv(projection_class, base):
            pass
        base = ProjectedEnv
    if settings.route_input:
        from atc.route_input import RouteInput
        input_class = RouteInput
        if settings.route_residual:
            from atc.residual import RouteResidualControl
            input_class = RouteResidualControl
        if settings.route_choice:
            from atc.route_choice import RouteChoiceInput, RouteChoiceResidual
            input_class = RouteChoiceResidual if settings.route_residual else RouteChoiceInput
        class RouteInputEnv(input_class, base):
            pass
        base = RouteInputEnv
    if settings.observation_traffic_slots is not None:
        from atc.traffic_slots import TrafficSlots
        class TrafficObservationEnv(TrafficSlots, base):
            pass
        base = TrafficObservationEnv
        kwargs["observation_traffic_slots"] = settings.observation_traffic_slots
    if settings.conflict_features:
        from atc.conflicts import ConflictPrediction
        class PredictiveEnv(ConflictPrediction, base):
            pass
        base = PredictiveEnv
    env = base(**kwargs)
    return gym.wrappers.FlattenObservation(env) if kind == "sa" else FlattenObservations(env)



def _make_multiagent_world(recipe, guard_static=False, guard_traffic=False):
    import supersuit as ss
    from atc.vector import FixedPopulation
    return ss.pettingzoo_env_to_vec_env_v1(FixedPopulation(make_env("ma", recipe, guard_static, guard_traffic)))


def make_training_env(kind, workers=1, recipe="baseline", guard_static=False, guard_traffic=False):
    if workers < 1:
        raise ValueError("workers must be positive")
    if kind == "sa":
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        constructors = [partial(make_env, kind, recipe, guard_static, guard_traffic) for _ in range(workers)]
        if workers == 1:
            return DummyVecEnv(constructors)
        return SubprocVecEnv(constructors, start_method="spawn")
    from supersuit.vector import MakeCPUAsyncConstructor
    from atc.vector import SeededVecEnv
    prototype = _make_multiagent_world(recipe, guard_static, guard_traffic)
    if workers == 1:
        return SeededVecEnv(prototype)
    # Construct the world inside its worker; pickling an initialized BlueSky world
    # would omit the process-global simulator when Python uses spawn.
    constructors = [partial(_make_multiagent_world, recipe, guard_static, guard_traffic) for _ in range(workers)]
    vec = MakeCPUAsyncConstructor(workers)(
        constructors, prototype.observation_space, prototype.action_space,
    )
    prototype.close()
    return SeededVecEnv(vec)
