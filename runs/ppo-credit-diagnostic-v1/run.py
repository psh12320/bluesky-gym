"""Record reward timing and critic predictions on preserved development trajectories."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import csv,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/onpolicy-position-source-v1-verify"
sys.path.insert(0,str(SOURCE))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ.update(SDL_VIDEODRIVER="dummy",PYGAME_HIDE_SUPPORT_PROMPT="1",PYTHONDONTWRITEBYTECODE="1")
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    import numpy as np
    import torch
    import bluesky as bs
    from atc_rl.cluster import verify
    from atc_rl.deployment import DecentralizedActor,make_environment,dictionary_offsets
    from atc.metrics import METRICS
    torch.set_num_threads(1)
    plan=read(WORK/"protocol.json")
    verify(SOURCE)
    simulator=WORK/"simulator"
    simulator.mkdir(exist_ok=False)
    os.chdir(WORK)
    bs.init(mode="sim",detached=True,workdir=str(simulator))
    results={}
    for case,setting in plan["cases"].items():
        checkpoint=ROOT/setting["checkpoint"]
        assert sha(checkpoint)==setting["sha256"]
        actor=DecentralizedActor.load(checkpoint)
        assert actor.configuration["algorithm"]=="ppo" and actor.configuration["progress_scale"]==0
        assert actor.configuration["reward_scale"]==.01 and actor.configuration.get("exploration","gaussian")=="gaussian"
        env=make_environment("ma",actor)
        world=env.unwrapped
        offsets,_=dictionary_offsets(world.observation_space(env.possible_agents[0]))
        rows=[];terminals=[]
        try:
            for episode in range(plan["worlds"]):
                obs,infos=env.reset(seed=plan["seed"] if episode==0 else None)
                scenario=hashlib.sha256(json.dumps(asdict(world.scenario),sort_keys=True,separators=(",",":"),default=float).encode()).hexdigest()
                decision=0
                while env.agents:
                    active=list(env.agents)
                    before={a:dict(world.metrics[a]) for a in active}
                    local=np.stack([obs[a] for a in active]).astype(np.float32)
                    inputs={"actor":torch.as_tensor(local),"critic":torch.zeros((len(active),*actor._critic_shape))}
                    with torch.no_grad():
                        values=actor.model.policy.predict_values(inputs).numpy().reshape(-1)
                    actions={a:actor(obs[a]) for a in active}
                    following,rewards,terms,truncs,infos=env.step(actions)
                    for index,a in enumerate(active):
                        change={k:float(infos[a][k])-float(before[a][k]) for k in METRICS}
                        goal=world.reach_reward*change["waypoint_reached"]
                        intrusion=world.intrusion_penalty*change["intrusion_time"]
                        restricted=world.restricted_area_penalty*change["time_in_restricted_area"]
                        outside=world.sector_exit_penalty*change["time_outside_sector"]
                        drift=float(rewards[a])-goal-intrusion-restricted-outside
                        assert abs(change["total_reward"]-float(rewards[a]))<1e-8
                        assert -abs(world.drift_penalty)*np.pi-1e-8<=drift<=1e-8
                        predicted=local[index,offsets["traffic_predicted_conflict"]]>0
                        entries=local[index,offsets["traffic_entry_time"]]*180.
                        rows.append({"case":case,"episode":episode,"agent":a,"scenario_sha256":scenario,
                            "decision":decision,"start_seconds":decision*5,"terminal":int(terms[a] or truncs[a]),
                            "heading_action":float(actions[a][0]),"speed_action":float(actions[a][1]),
                            "critic_value":float(values[index]),"native_reward":float(rewards[a]),
                            "reward_goal":goal,"reward_intrusion":intrusion,"reward_restricted":restricted,
                            "reward_outside":outside,"reward_drift":drift,
                            "intrusion_seconds":change["intrusion_time"],"intrusion_events":change["intrusion_events"],
                            "predicted_threats":int(predicted.sum()),
                            "earliest_predicted_entry_seconds":float(entries[predicted].min()) if predicted.any() else None})
                        if terms[a] or truncs[a]:
                            terminals.append({"episode":episode,"agent":a,"scenario_sha256":scenario,**{k:float(infos[a][k]) for k in METRICS}})
                    obs=following;decision+=1
                    assert decision<=600
                print(json.dumps({"case":case,"world":episode,"decisions":decision}),flush=True)
        finally:env.close()
        reference=ROOT/setting["reference_csv"]
        assert sha(reference)==setting["reference_csv_sha256"]
        with reference.open(newline="",encoding="utf-8") as f:
            expected={(int(r["episode"]),r["agent"]):r for r in csv.DictReader(f) if int(r["episode"])<plan["worlds"]}
        assert len(terminals)==len(expected)==10*plan["worlds"]
        for row in terminals:
            target=expected[(row["episode"],row["agent"])]
            assert row["scenario_sha256"]==target["scenario_sha256"]
            for k in METRICS:assert row[k]==float(target[k]),(case,row["agent"],k,row[k],target[k])
        with (WORK/(case+"-steps.csv")).open("x",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with (WORK/(case+"-aircraft.csv")).open("x",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(terminals[0]));writer.writeheader();writer.writerows(terminals)
        assert sha(checkpoint)==setting["sha256"]
        results[case]={"live_decisions":len(rows),"metric_matches":len(terminals)*len(METRICS),
            "step_csv_sha256":sha(WORK/(case+"-steps.csv")),"checkpoint_sha256":setting["sha256"]}
    verify(SOURCE)
    with (WORK/"complete.json").open("x",encoding="utf-8") as f:
        json.dump({"cases":results,"completed_at_utc":datetime.now(timezone.utc).isoformat(),
            "diagnostic_only":True,"training_performed":False,"unseen_scenarios_used":False},f,indent=2)
    print(json.dumps({"complete":True,"cases":results}),flush=True)
if __name__=="__main__":main()
