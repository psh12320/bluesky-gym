"""Analyze all paired seeds of the fixed conflict-input experiment."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
plan=read(PARENT/"protocol.json")
sys.path.insert(0,plan["training_source"])
from atc_rl.audit import audit
from atc_rl.compare import load_evaluation
from atc_rl.checkpoint_identity import verified_checkpoint,policy_fingerprint
from atc_rl.algorithm_compare import initial_for
from atc_rl.cluster import verify
def main():
    destination=PARENT/"paired-three-seed-results.json"
    if destination.exists():raise FileExistsError(destination)
    verify(Path(plan["training_source"]))
    classical=load_evaluation(Path(plan["classical_evaluation"]))
    sources=None;evaluation_source=None;rows=[]
    rng=np.random.default_rng(701)
    indices=rng.integers(0,plan["evaluation_worlds"],size=(10000,plan["evaluation_worlds"]))
    for seed in plan["seeds"]:
        arms={}
        for arm in ("features","zeros"):
            folder=PARENT/f"seed-{seed}"/arm;training=folder/"train"
            checked=audit(training);config=read(training/"config.json")
            if checked["status"]!="complete" or not 100000<=checked["live_transitions"]<102560:
                raise ValueError("Incomplete or invalid budget")
            for key,value in {**plan["config"],**plan["arms"][arm],"seed":seed}.items():
                if config.get(key)!=value:raise ValueError("Recipe changed: "+key)
            provenance=read(training/"provenance.json")
            source={k:provenance[k] for k in ("source_sha256","python","packages")}
            if sources is None:sources=source
            if source!=sources:raise ValueError("Training source or environment differs")
            initial=load_evaluation(folder/"eval-initial-dev20");final=load_evaluation(folder/"eval-final-dev20")
            if initial[2]!=final[2] or initial[2]!=classical[2]:raise ValueError("Unpaired scenarios")
            for loaded in (initial,final):
                protocol=loaded[1]
                if protocol["seed"]!=20260 or protocol["episodes"]!=20:raise ValueError("Wrong development evaluation")
                if protocol.get("evaluation_reward_scale",1)!=1 or protocol.get("evaluation_progress_scale",0)!=0:
                    raise ValueError("Non-native evaluation rewards")
                for key in ("algorithm","guidance","filter","action_reference","conflict_features","mask_conflict_features"):
                    if protocol.get(key,False)!=config[key]:raise ValueError("Evaluation changed "+key)
                if evaluation_source is None:evaluation_source=protocol["source_sha256"]
                if protocol["source_sha256"]!=evaluation_source:raise ValueError("Evaluation implementation changed")
            checkpoint=verified_checkpoint(training,final[1]["checkpoint"])
            if final[1]["checkpoint"]["sha256"]!=checked["model_sha256"] or final[1]["checkpoint"]["live_transitions"]!=checked["live_transitions"]:
                raise ValueError("Wrong evaluated final checkpoint")
            own=initial_for(checkpoint,initial[1])
            curve=read(folder/"curve/curve.json")
            if len(curve["points"])!=3 or curve["points"][0]["live_transitions"]!=0 or not 50000<=curve["points"][1]["live_transitions"]<52560:
                raise ValueError("Incomplete registered learning curve")
            arms[arm]={"config":config,"audit":checked,"initial_fingerprint":policy_fingerprint(own),"initial":initial,"final":final,
                       "folder":folder,"curve":curve}
        left,right=arms["features"],arms["zeros"]
        ignored={"run_dir","mask_conflict_features","actor_information"}
        if {k:v for k,v in left["config"].items() if k not in ignored}!={k:v for k,v in right["config"].items() if k not in ignored}:
            raise ValueError("Paired training settings differ beyond the feature mask")
        if left["initial_fingerprint"]!=right["initial_fingerprint"]:raise ValueError("Paired initial tensors differ")
        for key in ("actor_parameter_count","critic_parameter_count"):
            if left["audit"][key]!=right["audit"][key]:raise ValueError("Network capacity differs")
        effects={};arm_records={}
        for metric in left["initial"][3]:
            if not np.array_equal(left["initial"][3][metric],right["initial"][3][metric]):
                raise ValueError("Initial evaluated behavior differs")
            gains={arm:values["final"][3][metric].astype(float)-values["initial"][3][metric].astype(float) for arm,values in arms.items()}
            contrasts={**{arm+"_learning":value for arm,value in gains.items()},"features_minus_zeros_learning":gains["features"]-gains["zeros"]}
            effects[metric]={name:{"mean":float(value.mean()),"paired_world_bootstrap_95_interval":np.quantile(value[indices].mean(axis=1),[.025,.975]).tolist()}
                             for name,value in contrasts.items()}
        for arm,values in arms.items():
            before={k:float(v.mean()) for k,v in values["initial"][3].items()}
            after={k:float(v.mean()) for k,v in values["final"][3].items()}
            gate=plan["advance_screen"]
            checks={"arrival_level":after["waypoint_reached"]>=gate["arrival_at_least"],
                    "arrival_change":after["waypoint_reached"]-before["waypoint_reached"]>=gate["arrival_change_at_least"],
                    "clean_change":after["clean_completion"]-before["clean_completion"]>=gate["clean_completion_change_at_least"],
                    **{metric:after[metric]-before[metric]<=gate["safety_time_changes_at_most"] for metric in gate["metrics"]}}
            arm_records[arm]={"audit":values["audit"],"initial_means":before,"final_means":after,
                              "initial_csv_sha256":values["initial"][0]["csv_sha256"],"final_csv_sha256":values["final"][0]["csv_sha256"],
                              "registered_screen":checks,"screen_passed":all(checks.values()),"curve":str(values["folder"]/"curve/curve.json")}
        rows.append({"seed":seed,"identical_initial_policy_fingerprint":left["initial_fingerprint"],"arms":arm_records,"effects":effects})
    aggregate={}
    for metric in rows[0]["effects"]:
        aggregate[metric]={}
        for name in rows[0]["effects"][metric]:
            values=np.array([row["effects"][metric][name]["mean"] for row in rows])
            aggregate[metric][name]={"mean":float(values.mean()),"sample_standard_deviation":float(values.std(ddof=1)),
                                    "per_seed":{str(row["seed"]):float(value) for row,value in zip(rows,values)}}
    result={"training_seeds":plan["seeds"],"all_six_registered_runs_included":True,"rows":rows,"aggregate":aggregate,
            "protocol_sha256":hashlib.sha256((PARENT/"protocol.json").read_bytes()).hexdigest(),
            "classical_means":{k:float(v.mean()) for k,v in classical[3].items()},
            "uncertainty":"Per-seed intervals resample twenty worlds; aggregate sample standard deviations describe the three training seeds. No aircraft-level independence assumption.",
            "limitations":plan["limitations"]}
    destination.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"results":str(destination),"aggregate":aggregate}))
if __name__=="__main__":main()
