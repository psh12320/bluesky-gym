"""Verify categorical action transport against the frozen native benchmark."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
read = lambda path: json.loads(Path(path).read_text(encoding="utf-8-sig"))
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
PLAN = read(OUT / "integration-protocol.json")
SOURCE = Path(PLAN["source"])
sys.path.insert(0, str(SOURCE))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.update(PYTHONDONTWRITEBYTECODE="1", SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def verify():
    from atc_rl.cluster import verify as verify_source
    verify_source(SOURCE)
    for filename, expected in PLAN["protected_files"].items():
        if sha(ROOT / filename) != expected:
            raise ValueError("Protected evidence changed: " + filename)
    if sha(PLAN["bundle"]) != PLAN["bundle_sha256"]:
        raise ValueError("Source archive changed")


def model_mapping():
    from atc_rl.maneuvers import ManeuverMapping
    return ManeuverMapping.from_schema(read(ROOT / PLAN["schema"]))


def vector():
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from atc.metrics import METRICS, summarize
    from atc_rl.policy import AircraftPolicy
    from atc_rl.world_pool import WorldPool
    from atc_rl.maneuvers import ManeuverVecEnv, initialize_route_choice
    torch.set_num_threads(1)
    directory = OUT / "vector"
    directory.mkdir(exist_ok=False)
    config = PLAN["configuration"]
    base = WorldPool(1, directory / "workers", **{k: v for k, v in config.items() if k != "algorithm"})
    mapping = model_mapping()
    env = ManeuverVecEnv(base, mapping)
    try:
        model = PPO(AircraftPolicy, env, seed=PLAN["model_seed"], device="cpu", n_steps=4,
                    batch_size=40, n_epochs=1, learning_rate=3e-5,
                    policy_kwargs={"centralized":False})
        initialize_route_choice(model, PLAN["preferred_category_probability"])
        model.save(directory / "initial-model.zip")
        env.seed(PLAN["scenario_seed"])
        obs = env.reset()
        episode = decisions = checked_commands = 0
        records = []
        scenario = env.reset_infos[0]["scenario_sha256"]
        with (directory / "aircraft.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode", "scenario_sha256", "agent", *METRICS])
            writer.writeheader()
            while episode < PLAN["worlds"]:
                local_inputs = {"actor":obs["actor"], "critic":np.zeros_like(obs["critic"])}
                choices = model.predict(local_inputs, deterministic=True)[0]
                np.testing.assert_array_equal(choices, np.tile([0,2], (10,1)))
                commands = mapping.decode(obs["actor"], choices)
                np.testing.assert_array_equal(commands, base.goal_actions(obs))
                checked_commands += len(commands)
                obs, rewards, dones, infos = env.step(choices)
                decisions += 1
                for i, info in enumerate(infos):
                    if info["inactive"]:
                        assert dones[i] and rewards[i] == 0
                    if info["aircraft_done"]:
                        assert dones[i] and not info["inactive"]
                        row = {"episode":episode, "scenario_sha256":scenario, "agent":f"KL00{i+1}", **info["metrics"]}
                        records.append(row)
                        writer.writerow(row)
                        stream.flush()
                if infos[0]["world_completed"]:
                    assert len([r for r in records if r["episode"] == episode]) == 10
                    episode += 1
                    scenario = env.reset_infos[0]["scenario_sha256"]
                    print(json.dumps({"vector_worlds_complete":episode}), flush=True)
        assert model.num_timesteps == 0 and model._n_updates == 0
        result = summarize(records, PLAN["worlds"], 10)
        result.update(rl_updates=0, live_training_transitions=0, world_decisions=decisions,
                      checked_benchmark_commands=checked_commands, runtime=base.runtime,
                      model_sha256=sha(directory / "initial-model.zip"), mapping=mapping.specification(),
                      interpretation="Implementation parity for an explicit route-choice prior, not learned performance.")
        write(directory / "summary.json", result)
    finally:
        env.close()


def native():
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from atc.metrics import METRICS, summarize
    from atc_rl.maneuvers import ManeuverActor
    from atc_rl.deployment import make_environment
    from atc_rl.competition import EpisodeTrace, TraceMA
    from scripts import evaluate_competition as harness
    torch.set_num_threads(1)
    directory = OUT / "native"
    directory.mkdir(exist_ok=False)
    model_path = OUT / "vector/initial-model.zip"
    assert sha(model_path) == read(OUT / "vector/summary.json")["model_sha256"]
    model = PPO.load(model_path, device="cpu")
    assert model.num_timesteps == 0 and model._n_updates == 0
    actor = ManeuverActor(model, model_mapping(), PLAN["configuration"])
    os.chdir(directory)
    import bluesky as bs
    bs.init(mode="sim", detached=True, workdir=str(directory / "simulator"))
    from bluesky.core.entity import getproxied
    backend = bs.tools.geo.kwikqdrdist.__module__
    performance = type(getproxied(bs.traf.perf)).__module__
    assert backend == "bluesky.tools.geo._cgeo" and "openap" in performance.lower()
    trace = EpisodeTrace("ma", directory)
    runtime = {}
    opened = []
    def make_env(kind, n_agents=10):
        assert kind == "ma" and n_agents == 10
        env = make_environment(kind, actor, n_agents)
        runtime.update(env.deployment_runtime, geo_backend=backend, performance_module=performance)
        wrapper = TraceMA(env, trace)
        opened.append(wrapper)
        return wrapper
    harness.make_env = make_env
    harness.SEED = PLAN["scenario_seed"]
    try:
        records = harness.run_multi_agent(PLAN["worlds"], actor, 10)
    finally:
        for env in opened:
            env.close()
        trace.close()
    assert len(records) == len(trace.records) == PLAN["worlds"] * 10
    assert all(float(a[k]) == b[k] for a,b in zip(records,trace.records) for k in METRICS)
    harness.write_csv(records, str(directory / "original-harness.csv"))
    result = summarize(trace.records, PLAN["worlds"], 10)
    result.update(rl_updates=0, actor_inference_uses_joint_context=False, runtime=runtime,
                  world_decisions=trace.decisions, model_sha256=sha(model_path), native_harness=True)
    write(directory / "summary.json", result)


def review():
    from atc.metrics import METRICS
    def rows(path):
        with path.open(newline="", encoding="utf-8") as stream:
            items = [r for r in csv.DictReader(stream) if int(r["episode"]) < PLAN["worlds"]]
        indexed = {(int(r["episode"]),r["scenario_sha256"],r["agent"]):r for r in items}
        assert len(indexed) == len(items) == PLAN["worlds"] * 10
        return indexed
    datasets = {name:rows(path) for name,path in {
        "classical":ROOT / PLAN["classical_csv"],
        "vector":OUT / "vector/aircraft.csv", "native":OUT / "native/aircraft.csv"}.items()}
    reference = datasets["classical"]
    for name in ("vector", "native"):
        assert datasets[name].keys() == reference.keys(), name
        for key in reference:
            for metric in METRICS:
                if float(datasets[name][key][metric]) != float(reference[key][metric]):
                    raise ValueError(f"{name} differs from fixed benchmark at {key}, {metric}")
    verify()
    result = {"status":"complete", "finished_at_utc":datetime.now(timezone.utc).isoformat(),
              "protocol_sha256":sha(OUT / "integration-protocol.json"),
              "script_sha256":sha(__file__), "exact_metric_values_per_comparison":len(reference)*len(METRICS),
              "comparisons":["categorical vector vs frozen classical", "categorical native vs frozen classical"],
              "vector":read(OUT / "vector/summary.json"), "native":read(OUT / "native/summary.json"),
              "evidence_scope":"Untrained navigation prior and action-transport parity only. No RL performance claim."}
    write(OUT / "integration-validation.json", result)
    print(json.dumps({k:result[k] for k in ("status","exact_metric_values_per_comparison","evidence_scope")}), flush=True)


def main():
    verify()
    if len(sys.argv) > 1:
        {"vector":vector, "native":native, "review":review}[sys.argv[1]]()
        return
    (OUT / "integration-started").mkdir(exist_ok=False)
    script_hash = sha(__file__)
    protocol_hash = sha(OUT / "integration-protocol.json")
    for stage in ("vector", "native", "review"):
        assert sha(__file__) == script_hash and sha(OUT / "integration-protocol.json") == protocol_hash
        with (OUT / f"{stage}.log").open("x", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable,"-u",__file__,stage], cwd=SOURCE,
                                    stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy())
        if result.returncode:
            raise RuntimeError("Integration stage failed; preserve " + stage + ".log")
        print(json.dumps({"stage_complete":stage}), flush=True)


if __name__ == "__main__":
    main()
