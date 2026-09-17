"""Check policy observation geometry against live simulator ground velocities."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from gymnasium import spaces
import bluesky as bs
from atc.conflicts import FEATURE_NAMES, conflict_features
from atc.envs import make_env

report = {"seed": 2026, "episodes_per_track": 1, "controller": "neutral",
          "purpose": "Observation adapter consistency, not policy quality or forecast accuracy",
          "tracks": {}}
for kind in ("ma", "sa"):
    env = make_env(kind, "public_cpa")
    maximum_position_error = maximum_velocity_error = maximum_feature_error = 0.0
    decisions = present_slots = absent_slots = predicted_slots = 0
    populations = set()
    observation, _ = env.reset(seed=2026)
    finished = False
    try:
        while not finished:
            world = env.unwrapped
            agents = list(env.agents) if kind == "ma" else [bs.traf.id[0]]
            populations.add(int(bs.traf.ntraf))
            assert bs.traf.wind.winddim == 0, "Ground/air velocity equivalence requires zero wind"
            for agent in agents:
                own = bs.traf.id2idx(agent)
                if kind == "ma":
                    raw = spaces.unflatten(world.observation_space(agent), observation[agent])
                else:
                    raw = spaces.unflatten(world.observation_space, observation)
                count = world.intruder_obs.n
                others = [i for i in range(bs.traf.ntraf) if i != own]
                distance = np.array([float(bs.tools.geo.kwikdist(bs.traf.lat[own], bs.traf.lon[own], bs.traf.lat[i], bs.traf.lon[i])) for i in others])
                order = [others[i] for i in np.argsort(distance)[:count]]
                position = np.zeros((count, 2))
                velocity = np.zeros((count, 2))
                for slot, other in enumerate(order):
                    bearing, nm = bs.tools.geo.kwikqdrdist(bs.traf.lat[own], bs.traf.lon[own], bs.traf.lat[other], bs.traf.lon[other])
                    angle = np.deg2rad(bearing)
                    position[slot] = float(nm) * 1852 * np.array([np.cos(angle), np.sin(angle)])
                    velocity[slot] = [bs.traf.gsnorth[other] - bs.traf.gsnorth[own], bs.traf.gseast[other] - bs.traf.gseast[own]]
                heading = np.deg2rad(bs.traf.hdg[own])
                inverse = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
                observed_position = np.column_stack((raw["x_r"], raw["y_r"])) * world.intruder_obs.pos_norm @ inverse
                observed_velocity = np.column_stack((raw["vx_r"], raw["vy_r"])) * world.intruder_obs.spd_norm @ inverse
                position_error = float(np.max(np.abs(observed_position - position)))
                velocity_error = float(np.max(np.abs(observed_velocity - velocity)))
                assert position_error < 1e-8, position_error
                assert velocity_error < 1e-10, velocity_error
                present = np.arange(count) < len(order)
                expected = conflict_features(position, velocity, present, world.intrusion_distance * 1852)
                for key in FEATURE_NAMES:
                    error = float(np.max(np.abs(raw[key] - expected[key])))
                    assert error < 1e-10, (key, error)
                    maximum_feature_error = max(maximum_feature_error, error)
                maximum_position_error = max(maximum_position_error, position_error)
                maximum_velocity_error = max(maximum_velocity_error, velocity_error)
                decisions += 1
                present_slots += int(np.sum(present))
                absent_slots += int(np.sum(~present))
                predicted_slots += int(np.sum(expected["traffic_predicted_conflict"]))
            if kind == "ma":
                observation, _, _, _, _ = env.step({agent: np.zeros(2) for agent in agents})
                finished = not env.agents
            else:
                observation, _, terminated, truncated, _ = env.step(np.zeros(2))
                finished = terminated or truncated
    finally:
        env.close()
    report["tracks"][kind] = dict(aircraft_decisions=decisions, present_slots=present_slots,
        absent_slots=absent_slots, predicted_conflict_slots=predicted_slots,
        traffic_populations=sorted(populations), max_position_error_m=maximum_position_error,
        max_velocity_error_mps=maximum_velocity_error, max_feature_error=maximum_feature_error,
        no_wind_verified=True, passed=True)
    print(json.dumps({kind: report["tracks"][kind]}), flush=True)
(Path(__file__).parent / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
