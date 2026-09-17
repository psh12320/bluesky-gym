"""Collect supervised labels from the unchanged classical benchmark's executed actions."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os

import numpy as np

RESERVED_SEEDS = {42, 2026, 2027, 20260, 20301, 20302}
ARRAY_KEYS = {"actor", "teacher_action", "nominal_action", "intervened", "native_reward",
              "terminated", "truncated", "aircraft_index", "decision"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CaptureAction:
    """Delegate each command exactly once and record the original normalized value."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.values = {}

    def __getattr__(self, key):
        return getattr(self.wrapped, key)

    def clear(self):
        self.values.clear()

    def execute(self, aircraft, value):
        if aircraft in self.values:
            raise ValueError("Multiple commands for one aircraft in a decision")
        scalar = float(value)
        if not math.isfinite(scalar) or not -1 <= scalar <= 1:
            raise ValueError("Teacher command must be finite and normalized")
        self.values[aircraft] = scalar
        return self.wrapped.execute(aircraft, value)


def validate_world(arrays, actor_dim):
    if set(arrays) != ARRAY_KEYS:
        raise ValueError("Unexpected demonstration fields")
    actor = arrays["actor"]
    if actor.ndim != 2 or actor.shape[1] != actor_dim or not len(actor):
        raise ValueError("Invalid actor observations")
    n = len(actor)
    if not np.isfinite(actor).all() or np.any((actor[:, -1] < 0) | (actor[:, -1] > 1)):
        raise ValueError("Invalid actor values or remaining time")
    for name in ("teacher_action", "nominal_action"):
        actions = arrays[name]
        if actions.shape != (n, 2) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError("Invalid normalized action labels")
    for name in ARRAY_KEYS - {"actor", "teacher_action", "nominal_action"}:
        if arrays[name].shape != (n,):
            raise ValueError("Unaligned demonstration rows")
    if not np.isfinite(arrays["native_reward"]).all():
        raise ValueError("Invalid native rewards")
    for name in ("intervened", "terminated", "truncated"):
        if arrays[name].dtype != np.bool_:
            raise ValueError("Expected boolean demonstration flags")
    for name in ("aircraft_index", "decision"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError("Expected integer aircraft/decision indices")
    if np.any((arrays["aircraft_index"] < 0) | (arrays["aircraft_index"] >= 10)):
        raise ValueError("Invalid aircraft index")
    if np.any((arrays["decision"] < 0) | (arrays["decision"] >= 600)):
        raise ValueError("Invalid five-second decision index")
    identities = np.column_stack((arrays["decision"], arrays["aircraft_index"]))
    if len(np.unique(identities, axis=0)) != n:
        raise ValueError("Duplicate aircraft decision")
    for aircraft in range(10):
        rows = np.flatnonzero(arrays["aircraft_index"] == aircraft)
        if not len(rows) or not np.array_equal(arrays["decision"][rows], np.arange(len(rows))):
            raise ValueError("Missing, unordered or inactive aircraft samples")
        endings = arrays["terminated"][rows] | arrays["truncated"][rows]
        if endings.sum() != 1 or not endings[-1]:
            raise ValueError("Each aircraft must have one final terminal sample")
    return n


def load_demonstrations(directory, expected_role):
    """Load only complete train/validation data, never diagnostic/evaluation worlds."""
    directory = Path(directory).resolve()
    if expected_role not in {"train", "validation"}:
        raise ValueError("Learning accepts explicit train or validation data only")
    plan = read(directory / "protocol.json")
    completion = read(directory / "complete.json")
    if plan["role"] != expected_role or not completion["eligible_for_learning"]:
        raise ValueError("Dataset role is not eligible for this learning operation")
    if plan["seed"] in RESERVED_SEEDS or plan["reference"] is not None:
        raise ValueError("Evaluation worlds must never enter supervised training")
    if completion["protocol_sha256"] != sha256(directory / "protocol.json"):
        raise ValueError("Dataset protocol changed")
    if completion["status"] != "complete" or completion["worlds"] != plan["worlds"]:
        raise ValueError("Incomplete demonstration dataset")
    if sha256(directory / "space.npz") != completion["space_sha256"]:
        raise ValueError("Observation-space metadata changed")
    with np.load(directory / "space.npz", allow_pickle=False) as archive:
        low, high = archive["actor_low"].copy(), archive["actor_high"].copy()
    if low.ndim != 1 or low.shape != high.shape:
        raise ValueError("Invalid observation-space bounds")
    worlds = read(directory / "worlds.json")
    if len(worlds) != plan["worlds"] or [w["episode"] for w in worlds] != list(range(plan["worlds"])):
        raise ValueError("Missing or reordered demonstration worlds")
    if sha256(directory / "worlds.json") != completion["world_manifest_sha256"]:
        raise ValueError("World manifest changed")
    buffers = {key: [] for key in ARRAY_KEYS}
    world_indices = []
    scenarios = []
    for entry in worlds:
        path = (directory / entry["file"]).resolve()
        if path.parent != directory or sha256(path) != entry["sha256"]:
            raise ValueError("Unsafe or changed demonstration archive")
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        n = validate_world(data, len(low))
        if n != entry["live_transitions"]:
            raise ValueError("Demonstration transition count changed")
        for key in ARRAY_KEYS:
            buffers[key].append(data[key])
        world_indices.append(np.full(n, entry["episode"], dtype=np.int32))
        scenarios.append(entry["scenario_sha256"])
    combined = {key: np.concatenate(parts, axis=0) for key, parts in buffers.items()}
    if len(combined["actor"]) != completion["live_transitions"]:
        raise ValueError("Dataset experience count differs from completion report")
    return {
        "arrays": combined, "world_index": np.concatenate(world_indices),
        "scenarios": scenarios, "actor_low": low, "actor_high": high,
        "protocol": plan, "completion": completion, "directory": str(directory),
    }


def require_disjoint(training, validation):
    if training["protocol"]["seed"] == validation["protocol"]["seed"]:
        raise ValueError("Training and validation generator seeds overlap")
    if set(training["scenarios"]) & set(validation["scenarios"]):
        raise ValueError("Training and validation contain the same world")
    if not np.array_equal(training["actor_low"], validation["actor_low"]) or not np.array_equal(
            training["actor_high"], validation["actor_high"]):
        raise ValueError("Training and validation observation spaces differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worlds", type=int, required=True)
    parser.add_argument("--role", choices=("train", "validation", "parity"), required=True)
    parser.add_argument("--reference", type=Path, help="Classical evaluation for a diagnostic-only exact replay")
    args = parser.parse_args()
    if args.seed < 0 or args.worlds < 1:
        parser.error("Nonnegative seed and positive world count required")
    if args.role == "parity":
        if args.seed != 20260 or args.reference is None:
            parser.error("Parity checks use the development reference and are ineligible for learning")
    elif args.seed in RESERVED_SEEDS or args.reference is not None:
        parser.error("Training data must use fresh worlds, without an evaluation reference")
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    import platform
    import time
    from datetime import datetime, timezone
    from dataclasses import asdict
    from gymnasium import spaces
    import bluesky as bs

    root = Path(__file__).resolve().parents[1]
    hashes = {path.relative_to(root).as_posix(): sha256(path)
              for package in ("atc", "atc_rl", "core", "bluesky_gym", "bluesky_zoo")
              for path in (root / package).rglob("*.py")}
    hashes["pyproject.toml"] = sha256(root / "pyproject.toml")
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role, "seed": args.seed, "worlds": args.worlds,
        "reference": str(args.reference.resolve()) if args.reference else None,
        "source_sha256": hashes, "python": platform.python_version(),
        "teacher": "Frozen goal tracker at speed +1, route input, combined static/traffic filtering and decimal heading transport.",
        "label": "Actual normalized direct heading/speed commands passed to the simulator, after joint filtering.",
        "actor_information": "Local observation with CPA features and remaining-time fraction; no privileged teacher state or joint critic context.",
        "teacher_uses_privileged_state": True,
        "teacher_privileged_information": "The unchanged joint filter uses simulator positions, aircraft goals, performance limits and earlier commands in stable aircraft order.",
        "teacher_modified": False, "reinforcement_learning_updates": 0,
        "scenario_protocol": "Seed once, then continue the generator stream.",
        "eligibility": "Only complete fresh train/validation collections may enter supervised learning; parity data are always rejected.",
    }
    (directory / "protocol.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    runtime = directory / "simulator"
    runtime.mkdir()
    bs.init(mode="sim", detached=True, workdir=str(runtime))
    from atc.envs import make_env
    from atc.baselines import goal_tracker
    from atc.metrics import METRICS, summarize
    from atc.heading_transport import attach_decimal_heading
    from atc.recipes import RECIPES
    from atc_rl.world_worker import observation_recipe
    from atc_rl.action_support import verify_guard_selection
    RECIPES["demonstration_observation_interval5"] = observation_recipe(True, True)
    environment = attach_decimal_heading(
        make_env("ma", "demonstration_observation_interval5", guard_static=True, guard_traffic=True))
    world = environment.unwrapped
    verify_guard_selection(environment, {"filter": True})
    assert world.action_frequency == 5 and world.episode_time_limit == 3000
    assert world.intrusion_distance == 5 and world.distance_margin == 5
    heading = CaptureAction(world.heading_action)
    speed = CaptureAction(world.speed_action)
    world.heading_action, world.speed_action = heading, speed
    agents = list(environment.possible_agents)
    assert len(agents) == 10
    predict = goal_tracker(world.observation_space(agents[0]), speed_action=1.)
    original_space = environment.observation_space(agents[0])
    low = np.append(original_space.low, 0).astype(np.float32)
    high = np.append(original_space.high, 1).astype(np.float32)
    np.savez(directory / "space.npz", actor_low=low, actor_high=high)
    schema = []
    cursor = 0
    for name, space in world.observation_space(agents[0]).spaces.items():
        size = spaces.flatdim(space)
        schema.append({"name": name, "offset": cursor, "size": size})
        cursor += size
    assert cursor == len(low) - 1
    schema.append({"name": "time_remaining_fraction", "offset": cursor, "size": 1})
    (directory / "input-schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    reference = {}
    if args.reference:
        summary = read(args.reference / "summary.json")
        if sha256(args.reference / "aircraft.csv") != summary["csv_sha256"]:
            raise ValueError("Classical reference CSV changed")
        with (args.reference / "aircraft.csv").open(newline="", encoding="utf-8") as stream:
            reference = {(int(r["episode"]), r["agent"]): r for r in csv.DictReader(stream)
                         if int(r["episode"]) < args.worlds}
        if len(reference) != args.worlds * 10:
            raise ValueError("Classical reference has too few worlds")
    all_metrics, worlds, live_count, intervention_count, exact = [], [], 0, 0, 0
    start = time.monotonic()
    try:
        with (directory / "aircraft.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode", "scenario_sha256", "agent", *METRICS])
            writer.writeheader()
            for episode in range(args.worlds):
                observation, info = environment.reset(seed=args.seed if episode == 0 else None)
                scenario_json = json.dumps(asdict(world.scenario), sort_keys=True, separators=(",", ":"), default=float)
                scenario = hashlib.sha256(scenario_json.encode()).hexdigest()
                samples = {key: [] for key in ARRAY_KEYS}
                decision = 0
                while environment.agents:
                    active = list(environment.agents)
                    remaining = 1. - world.sim_time / world.episode_time_limit
                    actor = {agent: np.append(observation[agent], remaining).astype(np.float32) for agent in active}
                    # Match the benchmark's matrix call shape, including inactive zero slots.
                    batch = np.zeros((10, len(low) - 1), dtype=np.float32)
                    for index, agent in enumerate(agents):
                        if agent in active:
                            batch[index] = observation[agent]
                    nominal = predict(batch)
                    actions = {agent: nominal[agents.index(agent)] for agent in active}
                    heading.clear()
                    speed.clear()
                    next_observation, rewards, terminated, truncated, infos = environment.step(actions)
                    if set(heading.values) != set(active) or set(speed.values) != set(active):
                        raise ValueError("Recorded commands do not match the active population")
                    for agent in active:
                        diagnostic = world.last_projection[agent]
                        label = np.array([heading.values[agent], speed.values[agent]], dtype=np.float32)
                        if not np.allclose(label, [diagnostic["commanded_heading_turn_deg"] / 45.,
                                                  diagnostic["commanded_speed_action"]], rtol=0, atol=1e-7):
                            raise ValueError("Executed commands differ from joint-filter diagnostics")
                        values = {
                            "actor": actor[agent], "teacher_action": label, "nominal_action": actions[agent],
                            "intervened": bool(diagnostic["traffic_projection_intervened"]),
                            "native_reward": float(rewards[agent]), "terminated": bool(terminated[agent]),
                            "truncated": bool(truncated[agent]), "aircraft_index": agents.index(agent),
                            "decision": decision,
                        }
                        for key in ARRAY_KEYS:
                            samples[key].append(values[key])
                        if terminated[agent] or truncated[agent]:
                            row = {"episode": episode, "scenario_sha256": scenario, "agent": agent,
                                   **{key: float(infos[agent][key]) for key in METRICS}}
                            if reference:
                                expected = reference[episode, agent]
                                if expected["scenario_sha256"] != scenario:
                                    raise ValueError("Demonstration replay changed the scenario")
                                for metric in METRICS:
                                    if row[metric] != float(expected[metric]):
                                        raise ValueError(f"Teacher replay changed {episode} {agent} {metric}")
                                    exact += 1
                            all_metrics.append(row)
                            writer.writerow(row)
                            stream.flush()
                    observation = next_observation
                    decision += 1
                    if decision > 600:
                        raise ValueError("World exceeded the competition deadline")
                arrays = {key: np.asarray(values, dtype=(
                    np.float32 if key in {"actor", "teacher_action", "nominal_action"} else
                    np.bool_ if key in {"intervened", "terminated", "truncated"} else
                    np.int32 if key in {"aircraft_index", "decision"} else np.float64))
                    for key, values in samples.items()}
                n = validate_world(arrays, len(low))
                if len([r for r in all_metrics if r["episode"] == episode]) != 10:
                    raise ValueError("Incomplete teacher world")
                path = directory / f"world-{episode:05d}.npz"
                np.savez_compressed(path, **arrays)
                interventions = int(arrays["intervened"].sum())
                live_count += n
                intervention_count += interventions
                worlds.append({"episode": episode, "file": path.name, "sha256": sha256(path),
                               "scenario_sha256": scenario, "live_transitions": n,
                               "interventions": interventions})
                pending = directory / "worlds.pending.json"
                pending.write_text(json.dumps(worlds, indent=2), encoding="utf-8")
                pending.replace(directory / "worlds.json")
                print(json.dumps({"world": episode, "live_transitions": n, "interventions": interventions,
                                  "total_live_transitions": live_count}), flush=True)
    finally:
        environment.close()
    for name, expected in hashes.items():
        if sha256(root / name) != expected:
            raise ValueError("Source changed during collection: " + name)
    summary = summarize(all_metrics, args.worlds, 10)
    summary["csv_sha256"] = sha256(directory / "aircraft.csv")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    completion = {
        "status": "complete", "worlds": args.worlds, "live_transitions": live_count,
        "intervened_transitions": intervention_count, "actor_dim": len(low),
        "eligible_for_learning": args.role != "parity",
        "reinforcement_learning_updates": 0, "teacher_source_unchanged": True,
        "exact_reference_metric_matches": exact,
        "protocol_sha256": sha256(directory / "protocol.json"),
        "world_manifest_sha256": sha256(directory / "worlds.json"),
        "space_sha256": sha256(directory / "space.npz"),
        "wall_seconds": time.monotonic() - start,
    }
    (directory / "complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
