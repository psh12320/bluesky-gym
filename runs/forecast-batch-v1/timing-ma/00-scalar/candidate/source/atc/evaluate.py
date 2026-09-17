import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from atc.envs import make_env
from atc.metrics import METRICS, summarize
from atc.inference import predict_actions
from atc.provenance import capture
from atc.recipes import RECIPES


def evaluate(kind, episodes, seed, predict, recipe="baseline", guard_static=False, track_goal=False, guard_traffic=False, goal_speed_action=0.0):
    env = make_env(kind, recipe, guard_static, guard_traffic)
    if track_goal:
        from atc.baselines import goal_tracker
        dictionary = env.env.observation_space if kind == "sa" else env.env.observation_space(env.possible_agents[0])
        predict = goal_tracker(dictionary, speed_action=goal_speed_action)
    records = []
    started = time.perf_counter()
    decisions = 0
    residual_statistics = {}
    route_statistics = {}
    traffic_statistics = {}
    projection_statistics = {"actions": 0, "interventions": 0, "no_feasible_candidate": 0}
    try:
        for episode in range(episodes):
            # Seed once, then continue the scenario RNG stream, as in the official harness.
            observation, info = env.reset(seed=seed if episode == 0 else None)
            if kind == "sa":
                terminated = truncated = False
                while not (terminated or truncated):
                    observation, _, terminated, truncated, info = env.step(predict(observation))
                    decisions += 1
                records.append({"episode": episode, "agent": "KL001", **info})
            else:
                while env.agents:
                    agents = list(env.agents)
                    actions = predict(np.stack([observation[a] for a in agents]))
                    observation, _, terminated, truncated, infos = env.step(dict(zip(agents, actions)))
                    decisions += 1
                    for agent, info in infos.items():
                        if terminated[agent] or truncated[agent]:
                            records.append({"episode": episode, "agent": agent, **info})
            for key, value in getattr(env.unwrapped, "projection_statistics", {}).items():
                projection_statistics[key] += value
            for key, value in getattr(env.unwrapped, "residual_statistics", {}).items():
                residual_statistics[key] = residual_statistics.get(key, 0) + value
            for key, value in getattr(env.unwrapped, "route_input_statistics", {}).items():
                route_statistics[key] = route_statistics.get(key, 0) + value
            for key, value in getattr(env.unwrapped, "traffic_projection_statistics", {}).items():
                traffic_statistics[key] = traffic_statistics.get(key, 0) + value
            finished = [r for r in records if r["episode"] == episode]
            completion = np.mean([r["waypoint_reached"] for r in finished])
            intrusion = np.mean([r["intrusion_time"] for r in finished])
            print(f"Scenario {episode + 1}/{episodes}: completion={completion:.0%}, intrusion={intrusion:.1f}s", flush=True)
    finally:
        env.close()
    result = summarize(records, episodes, 1 if kind == "sa" else 10)
    result.update({"wall_seconds": time.perf_counter() - started,
                   "world_decisions": decisions, "seed": seed, "env": kind})
    if projection_statistics["actions"]:
        result["static_projection_statistics"] = projection_statistics
    if residual_statistics:
        result["residual_statistics"] = residual_statistics
    if route_statistics:
        result["route_input_statistics"] = route_statistics
    if traffic_statistics:
        result["traffic_projection_statistics"] = traffic_statistics
    return records, result


def main():
    parser = argparse.ArgumentParser(description="Evaluate objective air traffic metrics.")
    parser.add_argument("--env", choices=["sa", "ma"], required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--controller", choices=["neutral", "goal"], default="neutral",
                        help="Classical controller when no learned model is supplied")
    parser.add_argument("--goal-speed-action", type=float,
                        help="Constant normalized speed command in [-1, 1] for the classical goal controller; default 0")
    parser.add_argument("--recipe", choices=list(RECIPES), default="baseline")
    parser.add_argument("--guard-static", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--guard-traffic", action=argparse.BooleanOptionalAction, default=None,
                        help="Joint traffic/static correction of direct heading and speed commands")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--batch-inference", action="store_true", help="Reproduce historical MA development runs; official harness uses individual predictions")
    parser.add_argument("--out", type=Path, required=True, help="Output prefix for CSV and JSON")
    args = parser.parse_args()
    if args.guard_traffic and args.guard_static is False:
        parser.error("Joint traffic filtering includes static constraints")
    if args.guard_static and RECIPES[args.recipe].goal_relative:
        parser.error("--guard-static does not support goal-relative control")
    if args.batch_inference and (args.env != "ma" or not args.model or args.official):
        parser.error("Batched inference is only for learned MA development runs")
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.official and (args.episodes != 1000 or args.seed != 42):
        parser.error("official protocol requires --episodes 1000 --seed 42")
    if args.seed == 42 and not args.official:
        parser.error("seed 42 is reserved for the official 1000-scenario evaluation")
    if args.controller == "goal" and (args.model or RECIPES[args.recipe].route_guided or RECIPES[args.recipe].goal_relative or RECIPES[args.recipe].route_residual):
        parser.error("The classical goal controller requires direct controls and no model")
    if args.goal_speed_action is not None:
        if args.controller != "goal" or args.model:
            parser.error("--goal-speed-action requires the classical goal controller without a model")
        if not np.isfinite(args.goal_speed_action) or not -1 <= args.goal_speed_action <= 1:
            parser.error("--goal-speed-action must be finite and within [-1, 1]")
    training_config = {}
    if args.model:
        if not args.model.is_file() and Path(str(args.model) + ".zip").is_file():
            args.model = Path(str(args.model) + ".zip")
        if not args.model.is_file():
            parser.error("Model file does not exist")
        for parent in (args.model.parent, args.model.parent.parent):
            config_path = parent / "config.json"
            if config_path.is_file():
                training_config = json.loads(config_path.read_text(encoding="utf-8"))
                if args.guard_traffic is None:
                    args.guard_traffic = training_config.get("guard_traffic", False)
                if args.guard_static is None:
                    args.guard_static = training_config.get("guard_static", False)
                for key in ("env", "algorithm", "recipe"):
                    if training_config.get(key) != getattr(args, key):
                        parser.error(f"Model training {key}={training_config.get(key)} differs from evaluation; use matching flags")
                break
        else:
            parser.error("Keep the original config.json alongside the checkpoint")
        from atc.submission import _validate_configuration
        try:
            _validate_configuration(args.env, training_config)
        except ValueError as exc:
            parser.error(str(exc))
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO, SAC
        load_kwargs = {"buffer_size": 1} if args.algorithm == "sac" else {}
        model = {"ppo": PPO, "sac": SAC}[args.algorithm].load(args.model, device="cpu", **load_kwargs)
        predict = lambda obs: predict_actions(model, obs, batched=args.batch_inference)
    else:
        predict = lambda obs: np.zeros((*np.asarray(obs).shape[:-1], 2), dtype=np.float32)
    args.guard_traffic = bool(args.guard_traffic)
    args.guard_static = bool(args.guard_static or args.guard_traffic)
    if args.guard_traffic and (RECIPES[args.recipe].goal_relative or RECIPES[args.recipe].route_guided):
        parser.error("Joint traffic filtering requires direct heading/speed control")
    for suffix in (".csv", ".json"):
        if args.out.with_suffix(suffix).exists():
            parser.error("evaluation output exists; choose a new prefix")
    capture(args.out.parent / (args.out.name + "_provenance"))
    records, result = evaluate(args.env, args.episodes, args.seed, predict, args.recipe, args.guard_static,
                               track_goal=args.controller == "goal", guard_traffic=args.guard_traffic,
                               goal_speed_action=args.goal_speed_action if args.goal_speed_action is not None else 0.0)
    result.update({"policy": str(args.model) if args.model else args.controller,
                   "recipe": args.recipe, "guard_static": args.guard_static, "guard_traffic": args.guard_traffic, "official_protocol": args.official,
                   "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest() if args.model else None,
                   "inference_mode": ("batched" if args.batch_inference else "per_aircraft") if args.model else args.controller})
    if args.controller == "goal":
        result["goal_speed_action"] = args.goal_speed_action if args.goal_speed_action is not None else 0.0
    result.update(RECIPES[args.recipe].action_configuration())
    if RECIPES[args.recipe].route_guided:
        from atc.routes import ROUTE_REVISION
        result["route_revision"] = ROUTE_REVISION
        result["training_route_revision"] = training_config.get("route_revision", 1) if args.model else None
    if args.guard_static and not RECIPES[args.recipe].route_guided:
        from atc.projection import PROJECTION_REVISION, HORIZON_SECONDS, CLEARANCE_KM
        result.update(static_projection_revision=PROJECTION_REVISION,
                      static_projection_horizon_seconds=HORIZON_SECONDS,
                      static_projection_clearance_km=CLEARANCE_KM)
    if RECIPES[args.recipe].route_input:
        from atc.route_input import configuration as route_input_configuration
        result.update(route_input_configuration(RECIPES[args.recipe].route_choice))
    if RECIPES[args.recipe].route_residual:
        from atc.residual import configuration as residual_configuration
        result.update(residual_configuration(RECIPES[args.recipe].fast_speed_reference))
    if args.guard_traffic:
        from atc.traffic_projection import configuration
        result.update(configuration())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "agent", *METRICS])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in records)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
