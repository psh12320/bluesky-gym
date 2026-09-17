
"""Measure exploration commands and physical response; no training or model selection."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import csv,hashlib,json,os,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/onpolicy-native-source-v2-verify"
sys.path.insert(0,str(SOURCE))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ.update(SDL_VIDEODRIVER="dummy",PYGAME_HIDE_SUPPORT_PROMPT="1")
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
wrap=lambda x:(x+180.)%360.-180.

def main():
    from atc_rl.cluster import verify
    verify(SOURCE)
    plan=read(WORK/"protocol.json")
    model=ROOT/plan["initial_checkpoint"]
    assert hashlib.sha256(model.read_bytes()).hexdigest()==plan["initial_checkpoint_sha256"]
    from atc_rl.deployment import DecentralizedActor,make_environment
    actor=DecentralizedActor.load(model)
    assert actor.record["live_transitions"]==0
    import torch
    assert torch.count_nonzero(actor.model.policy.action_net.weight)==0
    assert torch.count_nonzero(actor.model.policy.action_net.bias)==0
    assert torch.allclose(actor.model.policy.log_std.exp(),torch.full((2,),.05))
    from atc_rl.goal_offset import heading_commands
    from atc.metrics import METRICS,summarize
    from gymnasium import spaces
    import bluesky as bs
    simulator=WORK/"simulator"
    simulator.mkdir(exist_ok=False)
    os.chdir(WORK)
    bs.init(mode="sim",detached=True,workdir=str(simulator))
    from bluesky.core.entity import getproxied
    assert bs.tools.geo.kwikqdrdist.__module__=="bluesky.tools.geo._cgeo"
    assert "openap" in type(getproxied(bs.traf.perf)).__module__.lower()
    noise=np.random.default_rng(plan["noise_seed"]).standard_normal((2,600,10,2))
    np.save(WORK/"standard-normal-draws.npy",noise)
    summary={}
    hashes_by_case={}
    for case,setting in plan["cases"].items():
        env=make_environment("ma",actor)
        world=env.unwrapped
        agents=list(env.possible_agents)
        indices={a:i for i,a in enumerate(agents)}
        positions={};cursor=0
        for key,space in world.observation_space(agents[0]).spaces.items():
            positions[key]=cursor;cursor+=spaces.flatdim(space)
        cos_i,sin_i=positions["cos_drift"],positions["sin_drift"]
        original_execute=world.heading_action.execute
        applied={}
        def execute(ac_id,action):
            applied[ac_id]=float(action)*45.
            return original_execute(ac_id,action)
        world.heading_action.execute=execute
        episode_records=[]
        scenario_hashes=[]
        fields=["case","episode","decision","agent","sim_time","latent_heading","latent_speed",
                "latent_heading_unclipped","requested_turn_degrees","applied_turn_degrees",
                "heading_before","reference_heading_before","heading_after","actual_turn_degrees",
                "drift_after_degrees","static_override","mapped_turn_saturated"]
        with (WORK/(case+"-steps.csv")).open("x",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            try:
                for episode in range(2):
                    obs,info=env.reset(seed=plan["scenario_seed"] if episode==0 else None)
                    scenario=json.dumps(asdict(world.scenario),sort_keys=True,separators=(",",":"),default=float)
                    scenario_hashes.append(hashlib.sha256(scenario.encode()).hexdigest())
                    decision=0
                    while env.agents:
                        active=list(env.agents)
                        latent={}
                        raw_heading={}
                        heading_before={}
                        reference={}
                        for a in active:
                            i=indices[a]
                            heading_step=decision//setting["heading_hold_decisions"]*setting["heading_hold_decisions"]
                            raw=noise[episode,heading_step,i,0]*setting["std"]
                            speed=noise[episode,decision,i,1]*setting["std"]
                            latent[a]=np.clip([raw,speed],-1,1).astype(np.float32)
                            raw_heading[a]=float(raw)
                            heading_before[a]=float(bs.traf.hdg[bs.traf.id2idx(a)])
                            reference[a]=float(wrap(heading_before[a]-np.rad2deg(np.arctan2(obs[a][sin_i],obs[a][cos_i]))))
                        commands=heading_commands([latent[a] for a in active],[obs[a] for a in active],cos_i,sin_i)
                        applied.clear()
                        following,rewards,terms,truncs,infos=env.step(latent)
                        assert set(applied)==set(active)
                        for a,command in zip(active,commands):
                            idx=bs.traf.id2idx(a)
                            after=float(bs.traf.hdg[idx]) if idx>=0 else None
                            # A terminal aircraft can have been deleted before step returns.
                            turn=None if after is None else float(wrap(after-heading_before[a]))
                            drift=float(np.rad2deg(np.arctan2(following[a][sin_i],following[a][cos_i])))
                            writer.writerow(dict(case=case,episode=episode,decision=decision,agent=a,sim_time=world.sim_time,
                                latent_heading=float(latent[a][0]),latent_speed=float(latent[a][1]),
                                latent_heading_unclipped=raw_heading[a],requested_turn_degrees=float(command[0])*45.,
                                applied_turn_degrees=applied[a],heading_before=heading_before[a],reference_heading_before=reference[a],
                                heading_after=after,actual_turn_degrees=turn,drift_after_degrees=drift,
                                static_override=int(abs(applied[a]-float(command[0])*45.)>1e-5),
                                mapped_turn_saturated=int(abs(float(command[0]))>=1-1e-6)))
                            if terms[a] or truncs[a]:
                                episode_records.append({"episode":episode,"scenario_sha256":scenario_hashes[-1],"agent":a,
                                    **{k:float(infos[a][k]) for k in METRICS}})
                        obs=following;decision+=1
                        assert decision<=600
                    f.flush()
                    print(json.dumps({"case":case,"episode":episode,"decisions":decision}),flush=True)
            finally:
                world.heading_action.execute=original_execute
                env.close()
        assert len(episode_records)==20
        hashes_by_case[case]=scenario_hashes
        with (WORK/(case+"-aircraft.csv")).open("x",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(episode_records[0]))
            writer.writeheader();writer.writerows(episode_records)
        summary[case]=summarize(episode_records,2,10)
    assert all(v==next(iter(hashes_by_case.values())) for v in hashes_by_case.values())
    # The zero-noise control must reproduce the first two saved initial-policy worlds.
    with (ROOT/plan["initial_reference_csv"]).open(newline="",encoding="utf-8") as f:
        reference={(int(r["episode"]),r["agent"]):r for r in csv.DictReader(f) if int(r["episode"])<2}
    with (WORK/"zero-aircraft.csv").open(newline="",encoding="utf-8") as f:
        observed=list(csv.DictReader(f))
    for row in observed:
        expected=reference[(int(row["episode"]),row["agent"])]
        assert row["scenario_sha256"]==expected["scenario_sha256"]
        for k in METRICS:assert float(row[k])==float(expected[k]),(k,row,expected)
    verify(SOURCE)
    assert hashlib.sha256(model.read_bytes()).hexdigest()==plan["initial_checkpoint_sha256"]
    record={"cases":summary,"paired_scenario_hashes":hashes_by_case,
            "zero_noise_reference_metric_matches":180,
            "source_bundle_sha256":plan["source_bundle_sha256"],
            "noise_draws_sha256":hashlib.sha256((WORK/"standard-normal-draws.npy").read_bytes()).hexdigest(),
            "diagnostic_only":True,"trained_policies":False,"unseen_or_official_scenarios_used":False,
            "completed_at_utc":datetime.now(timezone.utc).isoformat()}
    with (WORK/"complete.json").open("x",encoding="utf-8") as f:json.dump(record,f,indent=2)
    print(json.dumps({"complete":True,"zero_noise_reference_metric_matches":180,"diagnostic_only":True}),flush=True)

if __name__=="__main__":main()
