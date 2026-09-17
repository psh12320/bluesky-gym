import argparse
import csv
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from atc.envs import make_env
from atc.metrics import METRICS
from atc.inference import predict_actions
from atc.provenance import capture


def main():
    parser = argparse.ArgumentParser(description="Trace a development scenario and inspect policy failures.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True, help="Zero-based index in the development sequence")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-inference", action="store_true", help="Reproduce historical batched development evaluations")
    parser.add_argument("--skip-prefix-rollouts", action="store_true",
                        help="Advance preceding resets only; requires matching saved evaluation evidence")
    parser.add_argument("--reference", type=Path, help="Saved evaluation prefix for exact selected-scenario verification")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seed in {42, 2027} or args.episode < 0:
        parser.error("Use a development scenario, reserving final-test streams")
    if args.skip_prefix_rollouts and args.reference is None:
        parser.error("Skipping prefix rollouts requires --reference")
    for parent in (args.model.parent, args.model.parent.parent):
        config_path = parent / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            break
    else:
        parser.error("Model configuration was not found")
    kind = config["env"]
    if kind not in {"ma", "sa"}:
        parser.error("Unsupported environment in model configuration")
    if kind == "sa" and args.batch_inference:
        parser.error("Batched inference is only a historical MA option")
    expected_records = None
    if args.reference is not None:
        from atc.compare import load_evaluation
        metadata, records = load_evaluation(args.reference)
        if (metadata["env"] != kind or metadata["seed"] != args.seed
                or args.episode >= metadata["episodes"]
                or metadata.get("model_sha256") != hashlib.sha256(args.model.read_bytes()).hexdigest()
                or metadata.get("inference_mode") != ("batched" if args.batch_inference else "per_aircraft")):
            parser.error("Reference track, seed, episode, checkpoint or inference mode differs")
        expected_records = {r["agent"]: r for r in records if r["episode"] == args.episode}
    if args.out.exists():
        parser.error("Output directory already exists")
    import torch
    from stable_baselines3 import PPO, SAC
    import bluesky as bs
    from core.tools import kwikqdrdist
    torch.set_num_threads(1)
    kwargs = {"buffer_size": 1} if config["algorithm"] == "sac" else {}
    model = {"ppo": PPO, "sac": SAC}[config["algorithm"]].load(args.model, device="cpu", **kwargs)
    capture(args.out)
    env = make_env(kind, config["recipe"], config.get("guard_static", False), config.get("guard_traffic", False))
    rows, finals, traffic_rows = [], [], []
    rng_unchanged = True
    try:
        for episode in range(args.episode + 1):
            obs, _ = env.reset(seed=args.seed if episode == 0 else None)
            if args.skip_prefix_rollouts and episode < args.episode:
                if kind == "sa":
                    # Consume route setup before discarding this prefix scenario;
                    # queued ADDWPT commands must not attach to the next reset's traffic.
                    bs.stack.process()
                continue
            rng_before = repr(env.unwrapped._np_random.bit_generator.state)
            agents = list(env.agents) if kind == "ma" else [env.unwrapped.agent]
            while agents:
                observations = obs if kind == "ma" else {agents[0]: obs}
                actions = predict_actions(model, np.stack([observations[a] for a in agents]), batched=args.batch_inference)
                if episode == args.episode:
                    for agent, action in zip(agents, actions):
                        idx = bs.traf.id2idx(agent)
                        other = [i for i in range(bs.traf.ntraf) if i != idx]
                        near = min((kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.lat[i], bs.traf.lon[i])[1] for i in other), default=float("inf"))
                        nearest_agent = min(other, key=lambda i: kwikqdrdist(
                            bs.traf.lat[idx], bs.traf.lon[idx], bs.traf.lat[i], bs.traf.lon[i])[1]) if other else None
                        goal = env.unwrapped._goal[agent] if kind == "ma" else (env.unwrapped.goal_lat, env.unwrapped.goal_lon)
                        scenario_time = float(env.unwrapped.metrics[agent]["flight_time"])
                        if kind == "sa":
                            traffic_rows.extend(dict(time=scenario_time, agent=bs.traf.id[i],
                                lat=float(bs.traf.lat[i]), lon=float(bs.traf.lon[i]),
                                heading_deg=float(bs.traf.hdg[i]), cas_mps=float(bs.traf.cas[i]),
                                tas_mps=float(bs.traf.tas[i])) for i in other)
                        bearing, distance = kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], *goal)
                        route_data = {}
                        if hasattr(env.unwrapped, "_routes"):
                            from shapely.geometry import Point
                            world = env.unwrapped
                            position = world._xy((bs.traf.lat[idx], bs.traf.lon[idx]))
                            route_bearing, route_distance = world._route_reference(agent)
                            target = position + route_distance * np.array([np.sin(np.deg2rad(route_bearing)), np.cos(np.deg2rad(route_bearing))])
                            route_data = dict(route_target_index=int(np.argmin(np.linalg.norm(world._routes[agent] - target, axis=1))),
                                              route_target_distance_km=route_distance,
                                              route_clearance_km=float(world._planners[agent].clearance),
                                              route_target_east_km=float(target[0]),
                                              route_target_north_km=float(target[1]),
                                              route_target_in_static_domain=(bool(world._projection_domain.covers(Point(target)))
                                                  if getattr(world, "_projection_domain", None) is not None else None),
                                              in_route_domain=world._planners[agent].domain.covers(Point(position)))
                        rows.append(dict(time=scenario_time, agent=agent, **route_data,
                                         lat=float(bs.traf.lat[idx]), lon=float(bs.traf.lon[idx]),
                                         cas_mps=float(bs.traf.cas[idx]), tas_mps=float(bs.traf.tas[idx]),
                                         goal_distance_km=float(distance * 1.852),
                                         heading_error_deg=float((bs.traf.hdg[idx] - bearing + 180) % 360 - 180),
                                         heading_action=float(action[0]), speed_action=float(action[1]),
                                         nearest_agent=bs.traf.id[nearest_agent] if nearest_agent is not None else "",
                                         nearest_nm=float(near), outside=bool(env.unwrapped._outside_sector[agent]
                                             if kind == "ma" else env.unwrapped._outside_sector)))
                if kind == "ma":
                    obs, _, terms, truncs, infos = env.step(dict(zip(agents, actions)))
                else:
                    obs, _, terminated, truncated, info = env.step(actions[0])
                    terms, truncs, infos = {agents[0]: terminated}, {agents[0]: truncated}, {agents[0]: info}
                if episode == args.episode:
                    projection = getattr(env.unwrapped, "last_projection", {})
                    for row in rows[-len(agents):]:
                        row.update(projection.get(row["agent"], {}))
                    finals.extend(dict(agent=a, **info) for a, info in infos.items() if terms[a] or truncs[a])
                agents = list(env.agents) if kind == "ma" else ([] if terminated or truncated else [env.unwrapped.agent])
            rng_unchanged &= rng_before == repr(env.unwrapped._np_random.bit_generator.state)
            print(f"Replayed development scenario {episode + 1}/{args.episode + 1}", flush=True)
        scenario = asdict(env.unwrapped.scenario)
    finally:
        env.close()
    with (args.out / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if traffic_rows:
        with (args.out / "intruder_trajectories.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(traffic_rows[0]))
            writer.writeheader()
            writer.writerows(traffic_rows)
    report = {"env": kind, "episode": args.episode, "seed": args.seed, "model": str(args.model),
              "sampling_seconds": env.unwrapped.action_frequency, "final_metrics": finals,
              "prefix_mode": "resets_only" if args.skip_prefix_rollouts else "full_rollouts",
              "prefix_setup_commands_processed": bool(args.skip_prefix_rollouts and kind == "sa" and args.episode),
              "scenario_rng_unchanged_during_rollout": rng_unchanged,
              "inference_mode": "batched" if args.batch_inference else "per_aircraft",
              "mean_metrics": {k: float(np.mean([r[k] for r in finals])) for k in METRICS},
              "trajectory_statistics": {key: {"mean": float(np.mean([r[key] for r in rows])),
                                              "min": float(np.min([r[key] for r in rows])),
                                              "max": float(np.max([r[key] for r in rows]))}
                                        for key in ("cas_mps", "goal_distance_km", "heading_error_deg", "heading_action", "speed_action")}}
    if expected_records is not None:
        actual = {row["agent"]: row for row in finals}
        if actual.keys() != expected_records.keys():
            raise AssertionError("Diagnostic aircraft records differ from reference")
        differences = {key: max(abs(actual[agent][key] - expected_records[agent][key])
                                for agent in actual) for key in METRICS}
        report["reference_comparison"] = dict(reference=str(args.reference),
            matches_reference=not any(differences.values()), max_absolute_metric_differences=differences)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    if args.skip_prefix_rollouts and not rng_unchanged:
        raise AssertionError("Rollout advanced scenario RNG; reset-only prefix is not justified")
    if expected_records is not None and not report["reference_comparison"]["matches_reference"]:
        raise AssertionError("Diagnostic differs from saved evaluation; inspect summary and use full prefix rollouts")
    plot_trace(rows, scenario, args.out / "trajectories.png", traffic_rows)
    print(json.dumps({"mean_metrics": report["mean_metrics"], "trajectory_statistics": report["trajectory_statistics"]}, indent=2))


def plot_trace(rows, scenario, path, traffic_rows=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    center = scenario["center"]
    def xy(points):
        points = np.asarray(points)
        north = (points[:, 0] - center[0]) * 60 * 1.852
        east = (points[:, 1] - center[1]) * 60 * 1.852 * np.cos(np.deg2rad(center[0]))
        return np.column_stack((east, north))
    fig, (map_ax, speed_ax) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    map_ax.add_patch(Polygon(xy(scenario["sector"]), closed=True, fill=False, edgecolor="#334155", linewidth=2))
    for obstacle in scenario["obstacles"]:
        map_ax.add_patch(Polygon(xy(obstacle["vertices"]), closed=True, facecolor="#ef4444", alpha=0.2, edgecolor="#ef4444"))
    for i, agent in enumerate(sorted({r["agent"] for r in traffic_rows})):
        group = [r for r in traffic_rows if r["agent"] == agent]
        position = xy([(r["lat"], r["lon"]) for r in group])
        map_ax.plot(*position.T, color="#64748b", linewidth=.7, alpha=.3,
                    label="Scripted intruders" if i == 0 else None)
    for i, spec in enumerate(scenario["agents"]):
        group = [r for r in rows if r["agent"] == spec["ac_id"]]
        position = xy([(r["lat"], r["lon"]) for r in group])
        color = plt.get_cmap("tab10")(i)
        map_ax.plot(*position.T, color=color, linewidth=1.3, label=spec["ac_id"])
        start, goal = xy([spec["start"], spec["goal"]])
        map_ax.scatter(*start, color=color, s=18)
        map_ax.scatter(*goal, color=color, marker="x", s=45)
        speed_ax.plot([r["time"] for r in group], [r["cas_mps"] for r in group], color=color, linewidth=1.2)
    map_ax.set_aspect("equal")
    bounds = xy(scenario["sector"])
    margin = np.ptp(bounds, axis=0) * .05
    map_ax.set_xlim(bounds[:, 0].min() - margin[0], bounds[:, 0].max() + margin[0])
    map_ax.set_ylim(bounds[:, 1].min() - margin[1], bounds[:, 1].max() + margin[1])
    map_ax.legend(fontsize=7)
    map_ax.set(xlabel="East (km)", ylabel="North (km)", title="Flight paths; crosses mark goals")
    speed_ax.set(xlabel="Simulation time (s)", ylabel="Calibrated airspeed (m/s)", title="Speed selection")
    map_ax.grid(alpha=0.15)
    speed_ax.grid(alpha=0.15)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
