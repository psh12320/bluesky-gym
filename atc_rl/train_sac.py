"""Train a fresh shared SAC comparison on the PPO/MAPPO simulator adapter."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
import zipfile


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--live-steps", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=51800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--collection-steps", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=5000,
                        help="SB3 counted slots; actual live count at first update is recorded")
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="SAC gradient rounds after each vector decision, following warmup")
    parser.add_argument("--initial-action-std", type=float, default=.05)
    parser.add_argument("--entropy-initial", type=float, default=.01)
    parser.add_argument("--reward-scale", type=float, default=.01)
    parser.add_argument("--action-reference", choices=["direct", "goal_offset"], default="goal_offset")
    parser.add_argument("--guidance", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--static-filter", action="store_true")
    parser.add_argument("--conflict-features", action="store_true")
    parser.add_argument("--mask-conflict-features", action="store_true")
    parser.add_argument("--traffic-position-scale", type=float, default=1.)
    parser.add_argument("--checkpoint-live-steps", type=int, default=100000)
    parser.add_argument("--max-wall-seconds", type=float, default=0.)
    args = parser.parse_args()
    positive = (args.workers, args.live_steps, args.collection_steps, args.buffer_size,
                args.batch_size, args.gradient_steps, args.checkpoint_live_steps)
    if min(positive) < 1 or args.learning_starts < 0:
        parser.error("Training budgets and worker counts must be positive; warmup nonnegative")
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds < 0:
        parser.error("Wall limit must be finite and nonnegative")
    for name in ("initial_action_std", "entropy_initial", "reward_scale", "traffic_position_scale"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(name + " must be finite and positive")
    if args.initial_action_std > 1:
        parser.error("Initial latent action standard deviation must be <= 1")
    if args.mask_conflict_features and not args.conflict_features:
        parser.error("Masking requires conflict features")
    seeds = [args.seed + 10 * i for i in range(args.workers)]
    if args.seed < 0 or set(seeds) & {42, 2026, 2027, 20260, 20301, 20302}:
        parser.error("Reserved evaluation seed or invalid seed")
    args.run_dir = args.run_dir.resolve()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("Choose an empty run directory; historical replay is never resumed")
    return args


def main():
    args = arguments()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    import numpy as np
    import torch
    from stable_baselines3.common.logger import configure
    from atc_rl.sac_support import LocalAircraftInputs, SACAccounting, build_sac, verify_accounting
    from atc_rl.world_pool import WorldPool
    from atc_rl.reward_scale import ScaledLearningRewards
    from atc_rl.training_records import TrainingRecords
    torch.set_num_threads(1)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    files = [root / "pyproject.toml"]
    for package in ("atc", "atc_rl", "core", "bluesky_gym", "bluesky_zoo"):
        files.extend((root / package).rglob("*.py"))
    hashes = {}
    with zipfile.ZipFile(directory / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            data, name = path.read_bytes(), path.relative_to(root).as_posix()
            hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
    def write(name, value):
        (directory / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    write("provenance.json", {"source_sha256": hashes, "python": platform.python_version(),
        "platform": platform.platform(), "packages": {name: importlib.metadata.version(name)
        for name in ("torch", "stable-baselines3", "numpy", "gymnasium", "pettingzoo", "bluesky-simulator")}})
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(algorithm="sac", world_seeds=[args.seed + 10 * i for i in range(args.workers)],
        shared_actor=True, shared_critic=True, actor_widths=[128, 128], critic_widths=[256, 256],
        number_of_critics=2, activation="Tanh", critic_information="same local observation as actor and candidate action",
        actor_information="WorldPool local observation including remaining-time fraction; no joint critic state",
        gamma=.996508469331006, tau=.005, learning_rate=3e-4, decision_interval_seconds=5,
        reward_recipe="public_weights", progress_scale=0., neutral_action_mean=True,
        exploration="gaussian", sde_weight_std=None, sde_sample_freq=None,
        action_distribution="SAC tanh-squashed diagonal Gaussian; initial_action_std is pre-tanh latent std",
        warmup_distribution="Independent zero-mean Gaussian with initial_action_std, clipped to action bounds",
        finite_horizon="Arrival and actual 3000-second task deadline are terminal; collection cuts bootstrap",
        replay="Live aircraft only; handle_timeout_termination=False; fresh replay, no legacy files loaded",
        entropy_coefficient="auto_" + str(args.entropy_initial), target_entropy=-2.,
        optimizer_steps_definition="Critic optimizer steps; actor and entropy optimizer steps recorded separately",
        budget_unit="Live aircraft transitions; finish current collection chunk and record exact overshoot",
        resume_supported=False, traffic_position_normalization_m=1000000./args.traffic_position_scale,
        comparison_limits="SAC has two Q critics and a squashed stochastic policy; architecture and update budgets differ from PPO")
    write("config.json", config)
    accounting = SACAccounting()
    counts = {"critic": 0, "actor": 0, "entropy": 0}
    first_update = {}
    checkpoints, hooks = [], []
    environment = None
    start = time.perf_counter()
    def step_hook(name):
        def record(*_):
            counts[name] += 1
            if not first_update:
                first_update.update(live_transitions=accounting.live_transitions,
                                    counted_transitions=accounting.live_transitions + accounting.padded_transitions)
        return record
    def save(model, label):
        path = directory / (label + ".zip")
        temporary = directory / (label + ".pending.zip")
        model.save(temporary)
        temporary.replace(path)
        record = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  **verify_accounting(model, accounting), "optimizer_steps": counts["critic"],
                  "actor_optimizer_steps": counts["actor"], "entropy_optimizer_steps": counts["entropy"],
                  "gradient_rounds": model._n_updates}
        checkpoints.append(record)
        write("checkpoints.pending.json", checkpoints)
        (directory / "checkpoints.pending.json").replace(directory / "checkpoints.json")
    try:
        pool = WorldPool(args.workers, directory / "workers", guidance=args.guidance, filter=args.filter,
            static_filter=args.static_filter, action_reference=args.action_reference,
            conflict_features=args.conflict_features, mask_conflict_features=args.mask_conflict_features,
            traffic_position_scale=args.traffic_position_scale)
        environment = pool
        write("runtime.json", pool.runtime)
        environment = LocalAircraftInputs(pool)
        environment = ScaledLearningRewards(environment, args.reward_scale)
        model = build_sac(environment, seed=args.seed, device=args.device, buffer_size=args.buffer_size,
            learning_starts=args.learning_starts, batch_size=args.batch_size, gradient_steps=args.gradient_steps,
            initial_action_std=args.initial_action_std, entropy_initial=args.entropy_initial)
        model.set_logger(configure(str(directory), ["csv"]))
        for name, optimizer in (("critic", model.critic.optimizer), ("actor", model.actor.optimizer),
                                ("entropy", model.ent_coef_optimizer)):
            hooks.append(optimizer.register_step_post_hook(step_hook(name)))
        initial = {name: parameter.detach().cpu().clone() for name, parameter in model.policy.named_parameters()}
        save(model, "initial-model")
        next_checkpoint = args.checkpoint_live_steps
        fields = ["collection", "live_transitions", "padded_transitions", "counted_transitions", "replay_size",
                  "world_decisions", "worlds_completed", "aircraft_completed", "optimizer_steps", "actor_optimizer_steps",
                  "entropy_optimizer_steps", "gradient_rounds", "wall_seconds", "live_transitions_per_second",
                  "training_return_last100", "native_return_last100", "training_arrival_last100",
                  "actor_loss", "critic_loss", "entropy_coefficient"]
        collection = 0
        with TrainingRecords(directory, args.reward_scale) as records, (directory / "learning.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            while accounting.live_transitions < args.live_steps:
                before = accounting.live_transitions
                model.learn(total_timesteps=args.collection_steps * environment.num_envs,
                            reset_num_timesteps=False, callback=accounting)
                if accounting.live_transitions <= before:
                    raise RuntimeError("Collection produced no live experience")
                collection += 1
                elapsed = time.perf_counter() - start
                completed = accounting.completed_aircraft[-100:]
                row = {"collection": collection, **verify_accounting(model, accounting),
                    "world_decisions": accounting.world_decisions, "worlds_completed": accounting.completed_worlds,
                    "aircraft_completed": len(accounting.completed_aircraft), "optimizer_steps": counts["critic"],
                    "actor_optimizer_steps": counts["actor"], "entropy_optimizer_steps": counts["entropy"],
                    "gradient_rounds": model._n_updates, "wall_seconds": elapsed,
                    "live_transitions_per_second": accounting.live_transitions / max(elapsed, 1e-9),
                    "training_return_last100": float(np.mean(accounting.completed_learning_returns[-100:])) if completed else None,
                    "native_return_last100": float(np.mean([r["total_reward"] for r in completed])) if completed else None,
                    "training_arrival_last100": float(np.mean([r["waypoint_reached"] for r in completed])) if completed else None,
                    "entropy_coefficient": float(model.log_ent_coef.detach().exp().cpu().item())}
                for name in ("actor_loss", "critic_loss"):
                    value = model.logger.name_to_value.get("train/" + name)
                    row[name] = float(value) if value is not None else None
                records.append(accounting.completed_aircraft, accounting.completed_learning_returns)
                writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps(row), flush=True)
                if accounting.live_transitions >= next_checkpoint:
                    save(model, f"policy-live-{accounting.live_transitions}")
                    next_checkpoint = (accounting.live_transitions // args.checkpoint_live_steps + 1) * args.checkpoint_live_steps
                if args.max_wall_seconds and elapsed >= args.max_wall_seconds:
                    break
        save(model, "model")
        changed = [name for name, parameter in model.policy.named_parameters()
                   if not torch.equal(initial[name], parameter.detach().cpu())]
        learned = any(name.startswith("actor.") for name in changed) and any(name.startswith("critic.") for name in changed)
        if model._n_updates and not learned:
            raise RuntimeError("Actor and critic did not both change despite optimizer updates")
        if any(not torch.isfinite(parameter).all() for parameter in model.policy.parameters()):
            raise RuntimeError("Nonfinite model parameters")
        if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
            raise RuntimeError("Source changed during training")
        if len(set(counts.values())) != 1 or counts["critic"] != model._n_updates:
            raise RuntimeError("SAC optimizer counts disagree")
        summary = {"status": "complete" if accounting.live_transitions >= args.live_steps else "wall_limit_before_budget",
            "algorithm": "sac", **verify_accounting(model, accounting), "optimizer_steps": counts["critic"],
            "actor_optimizer_steps": counts["actor"], "entropy_optimizer_steps": counts["entropy"],
            "gradient_rounds": model._n_updates, "first_update": first_update,
            "worlds_completed": accounting.completed_worlds, "aircraft_completed": len(accounting.completed_aircraft),
            "wall_seconds": time.perf_counter() - start, "actor_and_critic_parameters_changed": learned,
            "source_unchanged": True, "model_sha256": checkpoints[-1]["sha256"], "training_seed": args.seed,
            "performance_evaluation": "Separate paired initial/trained/classical evaluations required"}
        write("training_summary.json", summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write("failure.json", {"error": repr(error), "live_transitions": accounting.live_transitions,
                               "records_retained": True})
        raise
    finally:
        for hook in hooks:
            hook.remove()
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    main()
