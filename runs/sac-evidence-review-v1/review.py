"""Audit historical SAC from CSV, JSON and archived source, without loading replay files."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    from atc.metrics import METRICS,summarize
    original=ROOT/"runs/replication-frozen-v1"
    saved=read(original/"summary.json")
    records={}
    def numeric(rows):
        return [{**row,"episode":int(row["episode"]),**{k:float(row[k]) for k in METRICS}} for row in rows]
    for track in ("ma","sa"):
        prefix=ROOT/("runs/heldout-2027-ma-interval5-v1" if track=="ma" else "runs/heldout-2027-sa-reach250-v1")
        reference=prefix/"classical.csv"
        assert sha(reference)==saved["tracks"][track]["classical_csv_sha256"]
        with reference.open(newline="",encoding="utf-8") as f:base=list(csv.DictReader(f))
        baseline=summarize(numeric(base),200,10 if track=="ma" else 1)
        rows={}
        for seed in (2900,2902,2904):
            parent=prefix if seed==2900 else original/f"{track}-seed{seed}"
            stem="learned" if seed==2900 else "replication-200"
            path=parent/(stem+".csv");meta=read(parent/(stem+".json"))
            selected=saved["tracks"][track]["training_seeds"][str(seed)]
            assert sha(path)==selected["csv_sha256"]
            assert meta["model_sha256"]==selected["model_sha256"]
            with path.open(newline="",encoding="utf-8") as f:raw=list(csv.DictReader(f))
            result=summarize(numeric(raw),200,10 if track=="ma" else 1)
            assert meta["env"]==track and meta["seed"]==2027
            expected={k:result["metrics"][k]["mean"] if k in METRICS else result[k] for k in selected["values"]}
            assert expected==selected["values"]
            rows[str(seed)]={"csv_sha256":sha(path),"model_sha256":meta["model_sha256"],"metrics":expected,
                "trained_minus_classical":{k:value-(baseline["metrics"][k]["mean"] if k in METRICS else baseline[k]) for k,value in expected.items()}}
        aggregate={}
        for metric in next(iter(rows.values()))["metrics"]:
            changes=np.array([row["trained_minus_classical"][metric] for row in rows.values()])
            aggregate[metric]={"mean_change":float(changes.mean()),"sample_standard_deviation":float(changes.std(ddof=1)),
                               "per_seed":{seed:row["trained_minus_classical"][metric] for seed,row in rows.items()}}
        records[track]={"classical":baseline,"seeds":rows,"aggregate_trained_minus_classical":aggregate}
    settings={}
    for name in ("sac-ma-fast-reference-v1-25k-seed2900","sac-ma-fast-reference-v1-25k-seed2902",
                 "sac-ma-fast-reference-v1-25k-seed2904","sac-ma-public-learner1m-seed2500"):
        folder=ROOT/"runs"/name
        summary=read(folder/"training_summary.json")
        with zipfile.ZipFile(folder/"source.zip") as archive:
            candidates=[n for n in archive.namelist() if n.endswith("atc/replay.py")]
            assert len(candidates)==1
            archived_code=archive.read(candidates[0])
        assert hashlib.sha256(archived_code).hexdigest()==sha(ROOT/"atc/replay.py")
        with zipfile.ZipFile(folder/"model.zip") as archive:
            data=json.loads(archive.read("data"))
        assert data["num_timesteps"]==summary["timesteps"]
        settings[name]={"model_sha256":sha(folder/"model.zip"),
            "archived_replay_code_sha256":hashlib.sha256(archived_code).hexdigest(),
            "counted_transitions":data["num_timesteps"],"live_transitions":summary["live_transitions"],
            "skipped_transitions":summary["skipped_transitions"],"optimizer_updates":data["_n_updates"],
            "gamma":data["gamma"],"learning_starts":data["learning_starts"],
            "replay_loaded":False,"observed_training_timeout_count":None}
    # Reproduce only the archived replay class's behavior on synthetic inputs.
    from gymnasium.spaces import Box
    from atc.replay import AircraftReplayBuffer
    buffer=AircraftReplayBuffer(8,Box(-1,1,(1,),dtype=np.float32),Box(-1,1,(1,),dtype=np.float32),n_envs=2,device="cpu")
    x=np.zeros((2,1),dtype=np.float32)
    buffer.add(x,x,x,np.ones(2),np.ones(2,dtype=bool),
               [{"aircraft_terminated":True,"aircraft_truncated":False},
                {"aircraft_terminated":False,"aircraft_truncated":True}])
    effective=buffer._get_samples(np.array([0,1])).dones.cpu().numpy().reshape(-1).tolist()
    assert effective==[1.,0.]
    result={"recorded_at_utc":datetime.now(timezone.utc).isoformat(),
        "replication_summary_sha256":sha(original/"summary.json"),"tracks":records,"saved_training_settings":settings,
        "deadline_semantics_probe":{"synthetic_only":True,"arrival_target_done":effective[0],
            "task_deadline_target_done":effective[1],"new_ppo_finite_deadline_semantics_match":False},
        "interpretation":["SAC is a historical comparison, not an equal-budget algorithm ranking against the new PPO/MAPPO collector.",
          "The six historical replicated evaluations use the already-used stream 2027, not new reserved streams 20301/20302.",
          "These trained-versus-classical contrasts do not replace trained-versus-own-initial comparisons.",
          "The archived replay implementation masks deadline terminal flags as generic timeouts. New PPO/MAPPO use a finite task horizon and time input.",
          "No replay file is loaded. Frequency and causal effect of deadline bootstrapping in historical training remain unmeasured.",
          "No source, checkpoint, replay or old evaluation changes. Any future finite-horizon SAC comparison must use a new run."],
        "approval_review":{"replay_deserialization_rejected":True,"safer_alternative":"CSV/JSON recomputation, archived-source hashing and a synthetic replay probe; no serialized replay objects loaded."}}
    with (OUT/"review.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"deadline_semantics_probe":result["deadline_semantics_probe"],
      "aggregate":{track:{k:data["aggregate_trained_minus_classical"][k] for k in
       ("waypoint_reached","flight_time","intrusion_time","time_in_restricted_area","clean_completion_rate")} for track,data in records.items()}}))
if __name__=="__main__":main()
