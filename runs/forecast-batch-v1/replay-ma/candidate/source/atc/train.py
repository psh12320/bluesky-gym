import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from atc.callbacks import StopBeforeDeadline
from atc.algorithms import NavigationSAC, initialize_navigation_actor

from atc.envs import make_training_env
from atc.provenance import capture
from atc.recipes import RECIPES
from atc.replay import AircraftReplayBuffer


def parse_net_arch(value):
    try:
        layers = [int(width) for width in value.lower().split("x")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use hidden widths such as 64x64 or 256x256") from exc
    if not layers or min(layers) < 1:
        raise argparse.ArgumentTypeError("Hidden widths must be positive")
    return layers


def main():
    parser = argparse.ArgumentParser(description="Train reproducible PPO and SAC baselines.")
    parser.add_argument("--env", choices=["sa", "ma"], required=True)
    parser.add_argument("--algorithm", choices=["ppo", "sac"], required=True)
    parser.add_argument("--recipe", choices=list(RECIPES), default="baseline")
    parser.add_argument("--guard-static", action=argparse.BooleanOptionalAction, default=None, help="Filter direct or route-relative turns using projected static clearance")
    parser.add_argument("--guard-traffic", action=argparse.BooleanOptionalAction, default=None,
                        help="Joint traffic/static correction of direct heading and speed commands")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--net-arch", type=parse_net_arch, help="New models: hidden widths, e.g. 256x256 (default 64x64)")
    parser.add_argument("--learning-rate", type=float, help="New models: constant learning rate (default 0.0003)")
    parser.add_argument("--tau", type=float, help="New SAC models: target-network update coefficient (default 0.005)")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=250000)
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--critic-warmup-updates", type=int, help="New navigation SAC models: hold actor parameters for this many initial critic updates")
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--resume", type=Path, help="Continue a checkpoint with a fresh scenario RNG stream")
    parser.add_argument("--resume-replay", type=Path, help="Optional SAC replay from the same checkpoint")
    parser.add_argument("--save-replay", action="store_true", help="Save SAC replay with checkpoints and final model")
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--max-wall-seconds", type=float, default=0, help="Stop training early and save before a job time limit")
    args = parser.parse_args()
    navigation_reference = RECIPES[args.recipe].route_guided or RECIPES[args.recipe].route_residual
    if args.critic_warmup_updates is not None:
        if args.critic_warmup_updates < 0 or args.algorithm != "sac" or not navigation_reference:
            parser.error("--critic-warmup-updates requires navigation SAC and a nonnegative count")
        if args.resume:
            parser.error("Resume retains the saved critic warmup; omit --critic-warmup-updates")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("--learning-rate must be finite and positive")
    if args.tau is not None and (not math.isfinite(args.tau) or not 0 < args.tau <= 1):
        parser.error("--tau must be in (0, 1]")
    if args.tau is not None and args.algorithm != "sac":
        parser.error("--tau is only available for SAC")
    if args.resume and any(value is not None for value in (args.net_arch, args.learning_rate, args.tau)):
        parser.error("Resume retains the saved architecture, learning rate and tau; omit these new-model options")
    if args.guard_traffic and args.guard_static is False:
        parser.error("Joint traffic filtering includes static constraints")
    if args.guard_traffic:
        args.guard_static = True
    if args.guard_static and RECIPES[args.recipe].goal_relative:
        parser.error("--guard-static does not support goal-relative control")
    if args.batch_size is None:
        args.batch_size = 128 if args.algorithm == "ppo" else 256
    if args.learning_starts < 0 or args.gradient_steps < 1 or args.buffer_size < 1 or args.max_wall_seconds < 0:
        parser.error("Invalid warmup, gradient count, buffer size or wall-clock budget")
    if args.resume_replay and (not args.resume or args.algorithm != "sac"):
        parser.error("--resume-replay requires --resume and SAC")
    if args.save_replay and args.algorithm != "sac":
        parser.error("--save-replay is only available for SAC")
    if args.resume:
        if not args.resume.is_file():
            parser.error("Resume checkpoint does not exist")
        for parent in (args.resume.parent, args.resume.parent.parent):
            config_path = parent / "config.json"
            if config_path.is_file():
                saved = json.loads(config_path.read_text(encoding="utf-8"))
                if args.resume_replay:
                    for key, value in RECIPES[args.recipe].action_configuration().items():
                        if saved.get(key) != value:
                            name = "Decision interval" if key == "decision_interval_seconds" else "Speed-command mapping"
                            parser.error(f"{name} changed; use fresh replay")
                if args.resume_replay and RECIPES[args.recipe].route_guided:
                    from atc.routes import ROUTE_REVISION
                    if saved.get("route_revision", 1) != ROUTE_REVISION:
                        parser.error("Route tracking changed; resume weights with fresh replay instead")
                if args.resume_replay and RECIPES[args.recipe].route_input:
                    from atc.route_input import configuration as route_input_configuration
                    if any(saved.get(key) != value for key, value in route_input_configuration(RECIPES[args.recipe].route_choice).items()):
                        parser.error("Route input changed; use fresh replay")
                if args.resume_replay and RECIPES[args.recipe].route_residual:
                    from atc.residual import configuration as residual_configuration
                    if any(saved.get(key) != value for key, value in residual_configuration(RECIPES[args.recipe].fast_speed_reference).items()):
                        parser.error("Route residual mapping changed; use fresh replay")
                saved_traffic = bool(saved.get("guard_traffic", False))
                if args.guard_traffic is None:
                    args.guard_traffic = saved_traffic
                if args.guard_traffic:
                    if args.guard_static is False:
                        parser.error("Joint traffic filtering includes static constraints")
                    args.guard_static = True
                if args.resume_replay and bool(args.guard_traffic) != saved_traffic:
                    parser.error("Changing the traffic filter requires fresh replay")
                if args.resume_replay and saved_traffic:
                    from atc.traffic_projection import configuration
                    if any(saved.get(key) != value for key, value in configuration().items()):
                        parser.error("Traffic projection changed; use fresh replay")
                saved_filter = bool(saved.get("guard_static", False))
                if args.resume_replay and saved_filter and not RECIPES[args.recipe].route_guided:
                    from atc.projection import PROJECTION_REVISION
                    if saved.get("static_projection_revision") != PROJECTION_REVISION:
                        parser.error("Static projection changed; use fresh replay")
                if args.guard_static is None:
                    args.guard_static = saved_filter
                if args.resume_replay and bool(args.guard_static) != saved_filter:
                    parser.error("Changing the static filter requires fresh replay; omit --resume-replay")
                for key in ("env", "algorithm", "recipe"):
                    if saved.get(key) != getattr(args, key):
                        parser.error(f"Resume configuration mismatch: {key}")
                break
        else:
            parser.error("Resume checkpoint must have its original config.json")
    args.guard_static = bool(args.guard_static)
    args.guard_traffic = bool(args.guard_traffic)
    if args.guard_traffic and (RECIPES[args.recipe].goal_relative or RECIPES[args.recipe].route_guided):
        parser.error("Joint traffic filtering requires direct heading/speed control")
    if args.resume_replay and not args.resume_replay.is_file():
        parser.error("Resume replay file does not exist")
    world_seeds = set(range(args.seed, args.seed + args.workers))
    if world_seeds & {42, 2026, 2027}:
        parser.error("Worker seeds must avoid official seed 42 and development seeds 2026/2027")
    if min(args.workers, args.steps, args.checkpoint_every, args.batch_size, args.rollout_steps) < 1:
        parser.error("workers, steps, checkpoint interval, and batch size must be positive")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("run directory is not empty; choose a new directory")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(1)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["run_dir"] = str(args.run_dir.resolve())
    config["torch"] = torch.__version__
    config["world_seeds"] = sorted(world_seeds)
    config["step_unit"] = "SB3 vector transitions, including inactive aircraft padding in MA"
    config["gamma"] = RECIPES[args.recipe].gamma
    config["reward_kwargs"] = RECIPES[args.recipe].reward_kwargs()
    config.update(RECIPES[args.recipe].action_configuration())
    if RECIPES[args.recipe].route_input:
        from atc.route_input import configuration as route_input_configuration
        config.update(route_input_configuration(RECIPES[args.recipe].route_choice))
    if RECIPES[args.recipe].conflict_features:
        from atc.conflicts import HORIZON_SECONDS
        config["conflict_prediction_horizon_seconds"] = HORIZON_SECONDS
    navigation_reference = RECIPES[args.recipe].route_guided or RECIPES[args.recipe].route_residual
    if RECIPES[args.recipe].route_residual:
        from atc.residual import configuration as residual_configuration
        config.update(residual_configuration(RECIPES[args.recipe].fast_speed_reference))
    config["navigation_initialization"] = bool(navigation_reference and args.algorithm == "sac" and not args.resume)
    config["navigation_warmup_std"] = 0.15 if navigation_reference and args.algorithm == "sac" else None
    if RECIPES[args.recipe].route_guided:
        from atc.routes import ROUTE_REVISION
        config["route_revision"] = ROUTE_REVISION
    if args.guard_static and not RECIPES[args.recipe].route_guided:
        from atc.projection import PROJECTION_REVISION, HORIZON_SECONDS, CLEARANCE_KM
        config.update(static_projection_revision=PROJECTION_REVISION,
                      static_projection_horizon_seconds=HORIZON_SECONDS,
                      static_projection_clearance_km=CLEARANCE_KM)
    if args.guard_traffic:
        from atc.traffic_projection import configuration
        config.update(configuration())
    capture(args.run_dir)
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = make_training_env(args.env, args.workers, args.recipe, args.guard_static, args.guard_traffic)
    started = time.perf_counter()
    try:
        common = dict(seed=args.seed, device=args.device, verbose=1,
                      gamma=config["gamma"], learning_rate=args.learning_rate if args.learning_rate is not None else 3e-4,
                      policy_kwargs={"net_arch": args.net_arch if args.net_arch is not None else [64, 64]})
        sac_class = NavigationSAC if navigation_reference else SAC
        if args.resume:
            model = {"ppo": PPO, "sac": sac_class}[args.algorithm].load(args.resume, env=env, device=args.device)
            model.set_random_seed(args.seed)
            if args.algorithm == "sac":
                if args.resume_replay:
                    model.load_replay_buffer(args.resume_replay)
                    model.learning_starts = model.num_timesteps
                else:
                    model.learning_starts = model.num_timesteps + args.learning_starts
                model.gradient_steps = args.gradient_steps
                model.batch_size = args.batch_size
            print(f"Resuming at {model.num_timesteps} transitions with fresh scenario seed {args.seed}", flush=True)
        elif args.algorithm == "ppo":
            model = PPO("MlpPolicy", env, n_steps=args.rollout_steps, batch_size=args.batch_size,
                        gae_lambda=0.95, ent_coef=0.0, **common)
        else:
            if navigation_reference:
                common["critic_warmup_updates"] = args.critic_warmup_updates or 0
            model = sac_class("MlpPolicy", env, buffer_size=args.buffer_size,
                        batch_size=args.batch_size, learning_starts=args.learning_starts,
                        train_freq=1, gradient_steps=args.gradient_steps,
                        tau=args.tau if args.tau is not None else 0.005,
                        replay_buffer_class=AircraftReplayBuffer if args.env == "ma" else None, **common)
        if args.algorithm == "sac" and navigation_reference and not args.resume:
            initialize_navigation_actor(model)
        model.set_logger(configure(str(args.run_dir), ["stdout", "csv"]))
        config["observation_shape"] = list(model.observation_space.shape)
        config["policy_parameters"] = sum(parameter.numel() for parameter in model.policy.parameters())
        config["trainable_policy_parameters"] = sum(parameter.numel() for parameter in model.policy.parameters() if parameter.requires_grad)
        extractor = model.actor.features_extractor if args.algorithm == "sac" else model.policy.features_extractor
        config["features_extractor"] = type(extractor).__name__
        if hasattr(extractor, "projection"):
            config["prediction_projection_trainable"] = extractor.projection.weight.requires_grad
        if RECIPES[args.recipe].observation_traffic_slots is not None:
            config["observed_traffic_slots"] = RECIPES[args.recipe].observation_traffic_slots
        config["vector_slots"] = env.num_envs
        config["actual_gamma"] = model.gamma
        config["actual_net_arch"] = model.policy.net_arch
        config["actual_learning_rate"] = float(model.lr_schedule(model._current_progress_remaining))
        config["actual_critic_warmup_updates"] = getattr(model, "critic_warmup_updates", 0)
        config["actual_tau"] = getattr(model, "tau", None)
        config["actual_gradient_steps"] = getattr(model, "gradient_steps", None)
        config["initial_timesteps"] = model.num_timesteps
        config["actual_buffer_size"] = getattr(model, "buffer_size", None)
        config["actual_learning_starts"] = getattr(model, "learning_starts", None)
        config["actual_batch_size"] = model.batch_size
        config["starting_live_transitions"] = getattr(getattr(model, "replay_buffer", None), "live_transitions", 0)
        config["starting_skipped_transitions"] = getattr(getattr(model, "replay_buffer", None), "skipped_transitions", 0)
        (args.run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if RECIPES[args.recipe].route_residual and args.algorithm == "sac" and not args.resume:
            model.save(args.run_dir / "initial-model")
        callback = CheckpointCallback(
            save_freq=max(args.checkpoint_every // env.num_envs, 1),
            save_path=str(args.run_dir / "checkpoints"), name_prefix="model", save_replay_buffer=args.save_replay, verbose=2,
        )
        model.learn(total_timesteps=args.steps,
                    callback=[callback, StopBeforeDeadline(args.max_wall_seconds)],
                    reset_num_timesteps=not bool(args.resume),
                    log_interval=40 if args.env == "ma" and args.algorithm == "sac" else 4)
        model.save(args.run_dir / "model")
        if args.save_replay:
            model.save_replay_buffer(args.run_dir / "replay.pkl")
        result = {"timesteps": model.num_timesteps, "new_timesteps": model.num_timesteps - config["initial_timesteps"], "wall_seconds": time.perf_counter() - started}
        if args.algorithm == "sac":
            result["critic_updates"] = model._n_updates
            result["held_actor_updates"] = min(model._n_updates, getattr(model, "critic_warmup_updates", 0))
            result["actor_updates"] = result["critic_updates"] - result["held_actor_updates"]
        if isinstance(getattr(model, "replay_buffer", None), AircraftReplayBuffer):
            result["live_transitions"] = model.replay_buffer.live_transitions
            result["skipped_transitions"] = model.replay_buffer.skipped_transitions
            result["new_live_transitions"] = result["live_transitions"] - config["starting_live_transitions"]
            result["new_skipped_transitions"] = result["skipped_transitions"] - config["starting_skipped_transitions"]
        (args.run_dir / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
