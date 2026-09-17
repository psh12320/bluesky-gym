"""A bounded action-component diagnostic; no model or controller is retrained."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/onpolicy-gsde-source-v1-verify"
TRAINING = ROOT / "runs/ppo-gsde-timing-pilot-v1/seed-51700/every12/train"
sys.path.insert(0, str(SOURCE))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import torch
from stable_baselines3 import PPO
from atc.metrics import METRICS, SAFETY, summarize
from atc_rl.cluster import verify
from atc_rl.checkpoint_identity import verified_checkpoint
from atc_rl.world_pool import WorldPool

read = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def all_twenty(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 200
    return {(int(row["episode"]), row["agent"]): row for row in rows}


def main():
    torch.set_num_threads(1)
    verify(SOURCE)
    record = next(r for r in read(TRAINING / "checkpoints.json") if r["file"] == "model.zip")
    checkpoint = verified_checkpoint(TRAINING, record)
    model = PPO.load(checkpoint, device="cpu")
    config = read(TRAINING / "config.json")
    from atc_rl.exploration import validate_model
    validate_model(model, config)
    assert not model.policy.centralized
    assert config['guidance'] and config['static_filter'] and config['conflict_features'] and not config['filter']
    assert config['action_reference']=='goal_offset' and config['exploration']=='gsde' and config['sde_sample_freq']==12
    assert record['sha256']=='5fdebf1b43623b623b6575f4517b6035523a557a27cffaa26131302a5aa5d571'
    protected = [checkpoint, TRAINING/'config.json', TRAINING/'checkpoints.json',
                 ROOT/'runs/onpolicy-gsde-source-v1.zip', Path(__file__)]
    for stage in ('initial','final'):
        protected.extend(TRAINING.parent/f'eval-{stage}-dev20'/name
                         for name in ('aircraft.csv','summary.json','protocol.json','scenarios.json'))
    fingerprints = {str(p):sha(p) for p in protected}
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "protected_inputs": fingerprints,
        "source_manifest_sha256": sha(SOURCE/'cluster-manifest.json'),
        "analysis": "All physical metrics; paired whole-world bootstrap, 10000 resamples seed701; report both components and their interaction.",
        "controls": "Full and zero-both must reproduce every aircraft metric and scenario in the original final and initial twenty-world evaluations exactly, before continuing.",
        "resource_policy": "Sequential arms with one simulator worker; existing three coordinators remain unchanged; at most four simulator workers combined.",
        "zero_action_meaning": "Zero heading follows unchanged goal/route guidance; zero speed removes the learned speed increment. Static filtering remains active.",
        "checkpoint": record, "source": str(SOURCE), "diagnostic_script_sha256": sha(Path(__file__)),
        "worlds_per_arm": 20, "stream": 20260,
        "arms": ["full_policy", "zero_both", "zero_heading", "zero_speed"],
        "mapping": "Zero the selected latent action after deterministic model prediction, before goal_offset mapping.",
        "support": {"guidance": True, "filter": False, "static_filter": True, "conflict_features": True},
        "reason": "The persistent-exploration pilot improved clean completion by 16 percentage points but increased restricted-area time; identify action-component effects.",
        "selection": "Outcome-selected exploratory every12 checkpoint, seed51700, 100438 live transitions; fresh three-seed timing replications remain separate.",
        "limits": [
            "Twenty reused development worlds and one outcome-selected training seed; no unseen generalization or across-seed inference.",
            "Heading and speed interact through subsequent states; this is a closed-loop policy-component ablation.",
            "Changing inference behavior is not additional RL training.",
            "These variants are not nominated competition candidates.",
            "Disabling a component changes subsequent observations; this does not estimate retraining with that action removed.",
            "Zero heading retains route support and may still turn the aircraft; this is not a constant-heading controller.",
            "World bootstrap intervals do not account for outcome-based checkpoint selection or training-seed variation.",
        ],
    }
    with (OUT / "protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2)
    all_records, results, scenario_sequences = {}, {}, {}
    replay_checks = {}
    for arm in protocol["arms"]:
        folder = OUT / arm
        folder.mkdir(exist_ok=False)
        env = WorldPool(1, folder / "workers", guidance=True, filter=False,
                        action_reference="goal_offset", conflict_features=True, static_filter=True)
        records, scenarios = [], []
        start = time.monotonic()
        try:
            assert env.observation_space == model.observation_space
            env.seed(20260)
            observation = env.reset()
            scenario = env.reset_infos[0]["scenario_sha256"]
            episode = decisions = 0
            with (folder / "aircraft.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["episode", "scenario_sha256", "agent", *METRICS])
                writer.writeheader()
                while episode < 20:
                    actions = []
                    for index in range(10):
                        local = {key: observation[key][index] for key in observation}
                        local["critic"] = np.zeros_like(local["critic"])
                        actions.append(model.predict(local, deterministic=True)[0])
                    actions = np.asarray(actions, dtype=np.float32)
                    if arm == "zero_both":
                        actions[:] = 0
                    elif arm == "zero_heading":
                        actions[:, 0] = 0
                    elif arm == "zero_speed":
                        actions[:, 1] = 0
                    if not np.isfinite(actions).all() or np.any(np.abs(actions)>1):
                        raise ValueError("Invalid policy action")
                    observation, rewards, dones, infos = env.step(actions)
                    decisions += 1
                    for index, info in enumerate(infos):
                        if info['inactive']:
                            assert dones[index] and rewards[index]==0
                        if info["aircraft_done"]:
                            assert dones[index] and not info['inactive'] and not info['TimeLimit.truncated']
                            row = {"episode": episode, "scenario_sha256": scenario,
                                   "agent": f"KL00{index+1}", **info["metrics"]}
                            writer.writerow(row)
                            stream.flush()
                            records.append(row)
                    if infos[0]["world_completed"]:
                        assert sum(r['episode']==episode for r in records)==10
                        print(json.dumps({'arm':arm,'episode':episode}),flush=True)
                        scenarios.append(scenario)
                        episode += 1
                        scenario = env.reset_infos[0]["scenario_sha256"]
                    if decisions > 12000 or time.monotonic() - start > 3600:
                        raise RuntimeError("Diagnostic exceeded its twenty-world bound; retain outputs")
        finally:
            env.close()
        assert len(records)==200 and len(scenarios)==20
        if arm in ('full_policy','zero_both'):
            stage='final' if arm=='full_policy' else 'initial'
            reference=TRAINING.parent/f'eval-{stage}-dev20'
            expected=all_twenty(reference/'aircraft.csv')
            exact=0
            for row in records:
                previous=expected[row['episode'],row['agent']]
                assert row['scenario_sha256']==previous['scenario_sha256']
                for metric in METRICS:
                    assert row[metric]==float(previous[metric]),(arm,row['episode'],row['agent'],metric)
                    exact+=1
            assert scenarios==[item['sha256'] for item in read(reference/'scenarios.json')]
            replay_checks[arm]={'exact_metrics':exact,'all_scenarios_identical':True,
                                'reference':str(reference),'reference_csv_sha256':sha(reference/'aircraft.csv')}
            with (folder/'replay-check.json').open('x',encoding='utf-8') as f:
                json.dump(replay_checks[arm],f,indent=2)
        result = summarize(records, 20, 10)
        result.update(arm=arm, checkpoint_sha256=record["sha256"],
                      scenarios=scenarios, world_decisions=decisions,
                      csv_sha256=sha(folder / "aircraft.csv"),
                      wall_seconds=time.monotonic() - start, diagnostic_only=True)
        with (folder / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        all_records[arm] = records
        results[arm] = result
        scenario_sequences[arm] = scenarios
        print(json.dumps({"arm_completed": arm, "clean_completion": result["clean_completion_rate"],
                          "metrics": {k: v["mean"] for k, v in result["metrics"].items()}}), flush=True)
    assert len({tuple(v) for v in scenario_sequences.values()}) == 1
    existing_path = TRAINING.parent / "eval-final-dev20/aircraft.csv"
    existing = all_twenty(existing_path)
    for row in all_records["full_policy"]:
        expected = existing[row["episode"], row["agent"]]
        assert row["scenario_sha256"] == expected["scenario_sha256"]
        for metric in METRICS:
            assert row[metric] == float(expected[metric]), (row["episode"], row["agent"], metric)
    initial_path = TRAINING.parent / "eval-initial-dev20/aircraft.csv"
    initial_raw = all_twenty(initial_path)
    initial = [{**row, "episode": episode, **{metric: float(row[metric]) for metric in METRICS}}
               for (episode, _), row in initial_raw.items()]
    results["own_initial_policy"] = summarize(initial, 20, 10)
    assert sha(checkpoint) == record["sha256"]
    verify(SOURCE)
    for name,digest in fingerprints.items():assert sha(Path(name))==digest,name
    def world_values(records):
        worlds={i:[r for r in records if r['episode']==i] for i in range(20)}
        assert all(len(rows)==10 for rows in worlds.values())
        clean=lambda r:bool(r['waypoint_reached'] and all(r[k]==0 for k in SAFETY))
        values={metric:np.array([np.mean([r[metric] for r in worlds[i]]) for i in range(20)]) for metric in METRICS}
        values['clean_completion']=np.array([np.mean([clean(r) for r in worlds[i]]) for i in range(20)])
        values['all_aircraft_clean_completion']=np.array([all(clean(r) for r in worlds[i]) for i in range(20)],dtype=float)
        return values
    world_data={name:world_values(rows) for name,rows in all_records.items()}
    indices=np.random.default_rng(701).integers(0,20,size=(10000,20))
    describe=lambda d:{'mean_difference':float(d.mean()),
        'paired_world_bootstrap_95_interval':np.quantile(d[indices].mean(axis=1),[.025,.975]).tolist()}
    pairs={'full_minus_zero':('full_policy','zero_both'),
           'heading_only_minus_zero':('zero_speed','zero_both'),
           'speed_only_minus_zero':('zero_heading','zero_both'),
           'heading_only_minus_full':('zero_speed','full_policy'),
           'speed_only_minus_full':('zero_heading','full_policy')}
    effects={label:{metric:describe(world_data[a][metric]-world_data[b][metric])
                   for metric in world_data[a]} for label,(a,b) in pairs.items()}
    interaction={metric:describe(world_data['full_policy'][metric]-world_data['zero_heading'][metric]
                 -world_data['zero_speed'][metric]+world_data['zero_both'][metric]) for metric in world_data['full_policy']}
    with (OUT / "results.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "purpose": "Saved-policy action-component intervention; no training or candidate promotion.",
            "full_policy_metrics_matched_to_existing_evaluation": 1800,
            "all_arms_use_identical_scenarios": True, "results": results,
            "paired_effects": effects, "full_minus_heading_minus_speed_plus_zero": interaction,
            "means":{arm:{metric:float(values.mean()) for metric,values in data.items()} for arm,data in world_data.items()},
            "replay_checks":replay_checks, "promoted_to_candidate":False,
            "diagnostic_protocol_sha256":sha(OUT/'protocol.json'),
            "finished_at_utc":datetime.now(timezone.utc).isoformat(),
            "limitations": protocol["limits"],
        }, stream, indent=2)


if __name__ == "__main__":
    main()
