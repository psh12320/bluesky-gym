"""Swap heading and speed between fixed pre-PPO and post-PPO policies."""
from pathlib import Path
from datetime import datetime, timezone
import csv,hashlib,json,os,sys,time
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
PLAN=read(OUT/"protocol.json")
SOURCE=Path(PLAN["source"])
sys.path.insert(0,str(SOURCE))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"


def main():
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from atc.metrics import METRICS,SAFETY,summarize
    from atc_rl.world_pool import WorldPool
    from atc_rl.cluster import verify
    from atc_rl.checkpoint_identity import verified_checkpoint,policy_fingerprint
    from atc_rl.exploration import validate_model
    from component_selector import combine_policy_components
    torch.set_num_threads(1)
    protocol_sha=sha(OUT/"protocol.json");script_sha=sha(__file__)
    def preserve():
        verify(SOURCE)
        assert sha(PLAN["source_bundle"])==PLAN["source_bundle_sha256"]
        assert sha(PLAN["cache"])==PLAN["cache_sha256"]
        assert sha(PLAN["component_helper"])==PLAN["component_helper_sha256"]
        assert sha(OUT/"protocol.json")==protocol_sha and sha(__file__)==script_sha
        for path,digest in PLAN["protected_files"].items():assert sha(path)==digest,path
    preserve();(OUT/"started").mkdir(exist_ok=False)
    training=Path(PLAN["training"]);config=read(training/"config.json")
    assert config["algorithm"]=="ppo" and config["exploration"]=="gaussian" and config.get("pretraining")
    assert all(config[k]==value for k,value in PLAN["support"].items())
    manifest=read(training/"checkpoints.json")
    models={};checkpoints={}
    for label,filename in PLAN["checkpoints"].items():
        record=next(r for r in manifest if r["file"]==filename)
        path=verified_checkpoint(training,record)
        models[label]=PPO.load(path,device="cpu");checkpoints[label]=record
        validate_model(models[label],config)
        assert not models[label].policy.centralized
    assert checkpoints["initial"]["live_transitions"]==0 and checkpoints["trained"]["live_transitions"]==101628
    assert policy_fingerprint(training/"initial-model.zip")==policy_fingerprint(training/"pretraining/model.zip")
    def load_rows(path,limit=20):
        with path.open(newline="",encoding="utf-8") as stream:rows=list(csv.DictReader(stream))
        result=[]
        for row in rows:
            if int(row["episode"])<limit:
                result.append({**row,"episode":int(row["episode"]),**{m:float(row[m]) for m in METRICS}})
        assert len(result)==limit*10 and len({(r["episode"],r["agent"]) for r in result})==len(result)
        return result
    original={name:load_rows(Path(path)/"aircraft.csv") for name,path in PLAN["control_evaluations"].items()}
    sequence=[next(r["scenario_sha256"] for r in original["initial"] if r["episode"]==i) for i in range(20)]
    assert sequence==[next(r["scenario_sha256"] for r in original["final"] if r["episode"]==i) for i in range(20)]
    collected={"initial":original["initial"],"trained":original["final"]};summaries={};replays={}
    environment=WorldPool(1,OUT/"workers",**PLAN["support"])
    try:
        assert environment.observation_space==models["initial"].observation_space==models["trained"].observation_space
        for arm,(heading_source,speed_source) in PLAN["arms"].items():
            preserve();folder=OUT/arm;folder.mkdir(exist_ok=False)
            worlds=PLAN["control_replay_worlds"] if arm.endswith("_replay") else PLAN["worlds"]
            environment.seed(PLAN["scenario_seed"])
            observation=environment.reset();scenario=environment.reset_infos[0]["scenario_sha256"]
            episode=decisions=command_count=0
            speed_sum=negative_speed=positive_speed=absolute_heading=0.
            records=[];start=time.monotonic()
            with (folder/"aircraft.csv").open("x",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=["episode","scenario_sha256","agent",*METRICS]);writer.writeheader()
                while episode<worlds:
                    assert scenario==sequence[episode]
                    prediction={label:[] for label in models}
                    for index in range(10):
                        local={key:observation[key][index] for key in observation}
                        local["critic"]=np.zeros_like(local["critic"])
                        for label,model in models.items():prediction[label].append(model.predict(local,deterministic=True)[0])
                    commands=combine_policy_components(prediction["initial"],prediction["trained"],heading_source,speed_source)
                    observation,reward,done,infos=environment.step(commands);decisions+=1
                    mask=np.array([not info["inactive"] for info in infos])
                    active=commands[mask];command_count+=len(active)
                    speed_sum+=float(active[:,1].astype(np.float64).sum())
                    negative_speed+=int((active[:,1]<0).sum());positive_speed+=int((active[:,1]>0).sum())
                    absolute_heading+=float(np.abs(active[:,0].astype(np.float64)).sum()*45)
                    for index,info in enumerate(infos):
                        if info["inactive"]:assert done[index] and reward[index]==0
                        if info["aircraft_done"]:
                            assert done[index] and not info["inactive"] and not info["TimeLimit.truncated"]
                            row={"episode":episode,"scenario_sha256":scenario,"agent":f"KL00{index+1}",**info["metrics"]}
                            writer.writerow(row);stream.flush();records.append(row)
                    if infos[0]["world_completed"]:
                        assert sum(r["episode"]==episode for r in records)==10
                        print(json.dumps({"arm":arm,"worlds_completed":episode+1}),flush=True)
                        episode+=1;scenario=environment.reset_infos[0]["scenario_sha256"]
                    if decisions>worlds*600:raise RuntimeError("Exceeded the fixed simulator decision bound")
            if arm.endswith("_replay"):
                label="initial" if arm=="initial_replay" else "final"
                reference={(r["episode"],r["agent"]):r for r in original[label] if r["episode"]<worlds}
                for row in records:
                    expected=reference[row["episode"],row["agent"]]
                    assert row["scenario_sha256"]==expected["scenario_sha256"]
                    for metric in METRICS:
                        if row[metric]!=expected[metric]:raise ValueError(f"Control mismatch: {arm}, {row['episode']}, {row['agent']}, {metric}")
                replays[arm]={"matched_metric_values":len(records)*len(METRICS),"scenarios_identical":True}
            else:collected[arm]=records
            summary=summarize(records,worlds,10)
            summary.update(arm=arm,component_sources={"heading":heading_source,"speed":speed_source},
                world_decisions=decisions,csv_sha256=sha(folder/"aircraft.csv"),wall_seconds=time.monotonic()-start,
                pre_filter_commands={"live_commands":command_count,"mean_speed_increment":speed_sum/command_count,
                    "negative_speed_fraction":negative_speed/command_count,"positive_speed_fraction":positive_speed/command_count,
                    "mean_absolute_heading_command_degrees":absolute_heading/command_count},
                runtime=environment.runtime,diagnostic_only=True)
            with (folder/"summary.json").open("x",encoding="utf-8") as f:json.dump(summary,f,indent=2)
            summaries[arm]=summary
            print(json.dumps({"arm_complete":arm,"clean_completion":summary["clean_completion_rate"]}),flush=True)
    finally:environment.close()
    def world_values(rows):
        groups={i:[r for r in rows if r["episode"]==i] for i in range(20)}
        assert all(len(group)==10 for group in groups.values())
        for i,group in groups.items():assert {r["scenario_sha256"] for r in group}=={sequence[i]}
        clean=lambda r:bool(r["waypoint_reached"] and all(r[k]==0 for k in SAFETY))
        values={metric:np.array([np.mean([r[metric] for r in groups[i]]) for i in range(20)]) for metric in METRICS}
        values["clean_completion"]=np.array([np.mean([clean(r) for r in groups[i]]) for i in range(20)])
        values["all_aircraft_clean_completion"]=np.array([all(clean(r) for r in groups[i]) for i in range(20)],dtype=float)
        return values
    values={name:world_values(rows) for name,rows in collected.items()}
    indices=np.random.default_rng(701).integers(0,20,size=(10000,20))
    describe=lambda d:{"mean_difference":float(d.mean()),"paired_world_bootstrap_95_interval":np.quantile(d[indices].mean(axis=1),[.025,.975]).tolist()}
    pairs={"full_learning":("trained","initial"),"heading_learning_only":("trained_heading","initial"),
           "speed_learning_only":("trained_speed","initial"),"heading_only_minus_full":("trained_heading","trained"),
           "speed_only_minus_full":("trained_speed","trained")}
    effects={label:{metric:describe(values[a][metric]-values[b][metric]) for metric in values[a]} for label,(a,b) in pairs.items()}
    interaction={metric:describe(values["trained"][metric]-values["trained_heading"][metric]-values["trained_speed"][metric]+values["initial"][metric]) for metric in values["trained"]}
    preserve()
    result={"status":"complete","finished_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":protocol_sha,"script_sha256":script_sha,
        "checkpoint_records":checkpoints,"control_replay_checks":replays,"all_twenty_worlds_paired":True,
        "means":{name:{metric:float(v.mean()) for metric,v in data.items()} for name,data in values.items()},
        "paired_effects":effects,"interaction":interaction,"new_evaluations":summaries,
        "original_full_controls_reused_unchanged":PLAN["control_evaluations"],"limitations":PLAN["limitations"],"candidate_promoted":False}
    with (OUT/"results.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"status":"complete","candidate_promoted":False}),flush=True)


if __name__=="__main__":
    try:main()
    except BaseException as error:
        with (OUT/"failure.json").open("x",encoding="utf-8") as f:json.dump({"error":repr(error),"artifacts_retained":True},f,indent=2)
        raise
