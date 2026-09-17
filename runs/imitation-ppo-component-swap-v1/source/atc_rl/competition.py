"""Run the original competition loops with shared PPO/MAPPO deployment hooks."""
from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import time

import gymnasium as gym
from pettingzoo.utils import BaseParallelWrapper


def validate_protocol(seed, episodes, official, heldout):
    if episodes < 1 or seed < 0:
        raise ValueError("Positive episode count and nonnegative seed required")
    if official and (seed != 42 or episodes != 1000):
        raise ValueError("Official evaluation requires 1000 episodes and seed 42")
    if seed == 42 and not official:
        raise ValueError("Seed 42 is reserved for the full official evaluation")
    if seed in (20301, 20302) and not heldout:
        raise ValueError("Reserved unseen stream requires explicit --heldout")


class EpisodeTrace:
    """Observe native terminal records without modifying rewards or infos."""
    def __init__(self, kind, directory=None):
        self.kind = kind
        self.stream = None
        self.writer = None
        if directory is not None:
            from atc.metrics import METRICS
            self.stream = (directory / "aircraft.csv").open("x", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.stream, fieldnames=["episode", "scenario_sha256", "agent", *METRICS])
            self.writer.writeheader()
            self.stream.flush()
        self.records = []
        self.scenarios = []
        self.decisions = 0

    def reset(self, env):
        import bluesky as bs
        scenario = json.dumps(asdict(env.unwrapped.scenario), sort_keys=True,
                              separators=(",", ":"), default=float)
        expected = 10 if self.kind == "ma" else 11
        if bs.traf.ntraf != expected:
            raise ValueError("Evaluation changed the simulated aircraft population")
        self.scenarios.append({"episode": len(self.scenarios),
            "sha256": hashlib.sha256(scenario.encode()).hexdigest(),
            "simulated_aircraft_at_reset": int(bs.traf.ntraf)})
        print(json.dumps({"episode_started": len(self.scenarios)}), flush=True)

    def terminal(self, agent, info):
        from atc.metrics import METRICS
        current = self.scenarios[-1]
        self.records.append({"episode": current["episode"], "scenario_sha256": current["sha256"],
                             "agent": agent, **{key: float(info[key]) for key in METRICS}})
        if self.writer is not None:
            self.writer.writerow(self.records[-1])
            self.stream.flush()

    def close(self):
        if self.stream is not None:
            self.stream.close()


class TraceMA(BaseParallelWrapper):
    def __init__(self, env, trace):
        super().__init__(env)
        self.trace = trace

    def reset(self, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        self.trace.reset(self.env)
        return result

    def step(self, actions):
        result = self.env.step(actions)
        _, _, terminations, truncations, infos = result
        self.trace.decisions += 1
        for agent, info in infos.items():
            if terminations[agent] or truncations[agent]:
                self.trace.terminal(agent, info)
        return result


class TraceSA(gym.Wrapper):
    def __init__(self, env, trace):
        super().__init__(env)
        self.trace = trace

    def reset(self, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        self.trace.reset(self.env)
        return result

    def step(self, action):
        result = self.env.step(action)
        _, _, terminated, truncated, info = result
        self.trace.decisions += 1
        if terminated or truncated:
            self.trace.terminal(self.unwrapped.agent, info)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["ma", "sa"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_protocol(args.seed, args.episodes, args.official, args.heldout)
    except ValueError as error:
        parser.error(str(error))
    directory, model_path = args.out.resolve(), args.model.resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error("Choose an empty output directory")
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    from atc_rl import deployment
    from scripts import evaluate_competition as harness
    from atc.metrics import METRICS, summarize

    harness.load_policy = deployment.load_policy
    actor = harness.load_policy(args.env, model_path)
    if args.official and actor.record["live_transitions"] <= 0:
        raise ValueError("Official RL results require a trained policy")
    # Development tests use separate seeded streams; official settings stay canonical.
    if not args.official:
        harness.SEED = args.seed
    assert harness.SEED == args.seed and harness.N_AGENTS_MA == 10
    source_paths = [root / "scripts/evaluate_competition.py"]
    for package in ("atc", "atc_rl", "core", "bluesky_gym", "bluesky_zoo"):
        source_paths.extend((root / package).rglob("*.py"))
    sources = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in source_paths}
    config = actor.configuration
    protocol = {key: config.get(key, False) for key in
                ("guidance", "filter", "static_filter", "conflict_features", "mask_conflict_features", "neutral_action_mean")}
    protocol.update(algorithm=config["algorithm"], action_reference=config.get("action_reference", "direct"),
        initial_action_std=config.get("initial_action_std", .6065306597126334),
        seed=args.seed, episodes=args.episodes, track=args.env, official_protocol=args.official,
        checkpoint=actor.record, model_path=str(model_path), source_sha256=sources,
        evaluation_progress_scale=0.0, evaluation_reward_scale=1.0,
        training_progress_scale=config.get("progress_scale", 0.0), training_reward_scale=config.get("reward_scale", 1.0),
        inference="One local observation at a time; zero joint critic input; deterministic actor.",
        scenario_protocol="Seed once, then continue the original generator stream.",
        implementation="Original scripts.evaluate_competition run loops; permitted environment/policy hooks replaced in memory.",
        total_reward_interpretation="Competitor-defined reward; not comparable across teams.")
    from atc_rl.exploration import configuration as exploration_configuration
    protocol.update(exploration_configuration(config))
    if config["algorithm"] == "sac":
        protocol.update(inference="One local observation at a time; deterministic SAC actor.",
                        action_distribution=config["action_distribution"])
    from atc_rl.traffic_scaling import position_scale
    protocol['traffic_position_scale']=position_scale(config)
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    simulator = directory / "simulator"
    simulator.mkdir(exist_ok=True)
    cache = root / "runs/simulator/cache/navdata.p"
    if cache.is_file():
        target = simulator / "cache/navdata.p"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(cache, target)
    os.chdir(directory)
    import bluesky as bs
    bs.init(mode="sim", detached=True, workdir=str(simulator))
    from bluesky.core.entity import getproxied
    backend = bs.tools.geo.kwikqdrdist.__module__
    performance = type(getproxied(bs.traf.perf)).__module__
    if backend != "bluesky.tools.geo._cgeo" or "openap" not in performance.lower():
        raise RuntimeError("Use the validated compiled geography and OpenAP dynamics")
    trace = EpisodeTrace(args.env, directory)
    runtime = {}
    opened = []

    def make_env(kind, n_agents=10):
        env = deployment.make_env(kind, n_agents)
        runtime.update(env.deployment_runtime, geo_backend=backend, performance_module=performance)
        wrapper = TraceMA(env, trace) if kind == "ma" else TraceSA(env, trace)
        opened.append(wrapper)
        return wrapper

    harness.make_env = make_env
    started = time.perf_counter()
    try:
        if args.env == "ma":
            native_records = harness.run_multi_agent(args.episodes, actor, 10)
        else:
            native_records = harness.run_single_agent(args.episodes, actor)
    except BaseException:
        for env in opened:
            env.close()
        raise
    finally:
        trace.close()
    if len(native_records) != len(trace.records):
        raise ValueError("Trace and original harness disagree on terminal record count")
    for native, tracked in zip(native_records, trace.records):
        if any(float(native[key]) != tracked[key] for key in METRICS):
            raise ValueError("Trace changed a native scoring record")
    harness.write_csv(native_records, str(directory / "original-harness.csv"))
    result = summarize(trace.records, args.episodes, 10 if args.env == "ma" else 1)
    result.update(algorithm=config["algorithm"], seed=args.seed, track=args.env, official_protocol=args.official,
                  runtime=runtime, actor_inference_uses_joint_context=False,
                  world_decisions=trace.decisions, wall_seconds=time.perf_counter()-started,
                  csv_sha256=hashlib.sha256((directory / "aircraft.csv").read_bytes()).hexdigest(),
                  original_harness_csv_sha256=hashlib.sha256((directory / "original-harness.csv").read_bytes()).hexdigest())
    for name, digest in sources.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Evaluation source changed: " + name)
    (directory / "scenarios.json").write_text(json.dumps(trace.scenarios, indent=2), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
