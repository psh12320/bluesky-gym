"""Audit paired PPO fine-tuning rates with a shared learned starting policy."""
from datetime import datetime, timezone
from pathlib import Path
import argparse, csv, hashlib, json, math, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
STAGES=("initial","50k","final")
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate_pair(left,right):
    if left["algorithm"]!=right["algorithm"] or left["algorithm"]!="ppo":raise ValueError("Expected two PPO arms")
    if left["learning_rate"]!=3e-4 or right["learning_rate"]!=3e-5:raise ValueError("Unexpected learning-rate contrast")
    if left["max_wall_seconds"]!=3600 or right["max_wall_seconds"]!=0:
        raise ValueError("Unexpected recovery resource policy")
    skip={"run_dir","learning_rate","max_wall_seconds"}
    if {k:v for k,v in left.items() if k not in skip}!={k:v for k,v in right.items() if k not in skip}:
        raise ValueError("Learning-rate comparison changed another setting")
    if not left.get("pretraining") or left["pretraining"]["initial_policy_is_untrained"]:
        raise ValueError("Both PPO arms must retain the same supervised history")

def paired_effect(first,second,indices):
    first,second=np.asarray(first,dtype=float),np.asarray(second,dtype=float)
    if first.ndim!=1 or first.shape!=second.shape or len(first)==0 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Expected paired finite world metrics")
    differences=first-second
    return {"mean_difference":float(differences.mean()),"paired_world_bootstrap_95_interval":np.quantile(differences[indices].mean(axis=1),[.025,.975]).tolist()}

def expected_source(directory,training=False):
    files=read(directory/"cluster-manifest.json")["files"]
    return {k:v for k,v in files.items() if (training and k=="pyproject.toml") or
            (k.endswith(".py") and k.split("/")[0] in {"atc","atc_rl","core","bluesky_gym","bluesky_zoo"})}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-initial",action="store_true")
    args=parser.parse_args()
    destination=OUT/("paired-initial-check.json" if args.check_initial else "paired-results.json")
    if destination.exists():raise FileExistsError(destination)
    plan=read(OUT/"protocol.json")
    for name,expected in plan["protected_interrupted_files"].items():
        if sha(ROOT/name)!=expected:raise ValueError("Interrupted attempt changed")
    sys.path.insert(0,plan["training_source"])
    from atc_rl.cluster import verify
    from atc_rl.audit import audit
    from atc_rl.checkpoint_identity import policy_fingerprint,verified_checkpoint
    from atc_rl.compare import load_evaluation
    baseline=Path(plan["baseline_directory"])
    control_plan=read(Path(plan["baseline_protocol"]))
    roots={"default":baseline,"smaller":OUT}
    training_sources={"default":Path(control_plan["source"]),"smaller":Path(plan["training_source"])}
    evaluation_source=Path(plan["evaluation_source"])
    for source in (*training_sources.values(),evaluation_source):verify(source)
    if sha(plan["training_bundle"])!=plan["training_bundle_sha256"] or sha(plan["evaluation_bundle"])!=plan["evaluation_bundle_sha256"]:
        raise ValueError("Source archives changed")
    if sha(plan["baseline_protocol"])!=plan["baseline_protocol_sha256"] or sha(baseline/"train/config.json")!=plan["baseline_configuration_sha256"]:
        raise ValueError("Control registration changed")
    configs={name:read(folder/"train/config.json") for name,folder in roots.items()}
    validate_pair(configs["default"],configs["smaller"])
    fingerprints={name:policy_fingerprint(folder/"train/initial-model.zip") for name,folder in roots.items()}
    expected_parent=policy_fingerprint(plan["pretrained_model"])
    if len(set(fingerprints.values()))!=1 or fingerprints["default"]!=expected_parent:
        raise ValueError("Full initial policy tensors or BC parent differ")
    if sha(plan["pretrained_model"])!=plan["pretrained_model_sha256"]:raise ValueError("BC parent archive changed")
    for name,folder in roots.items():
        initial=next(r for r in read(folder/"train/checkpoints.json") if r["file"]=="initial-model.zip")
        verified_checkpoint(folder/"train",initial)
        if any(initial[k]!=0 for k in ("live_transitions","counted_transitions","optimizer_steps")):
            raise ValueError("PPO initial checkpoint already contains RL experience")
        if read(folder/"train/provenance.json")["source_sha256"]!=expected_source(training_sources[name],True):
            raise ValueError("Training source differs from registered snapshot")
    initial_check={"full_initial_tensor_fingerprints":fingerprints,"bc_parent_fingerprint":expected_parent,
                   "only_optimizer_setting_changed":"learning_rate","resource_change":"recovery disables elapsed-time cutoff","rates":{"default":3e-4,"smaller":3e-5},
                   "supervised_initialization":configs["default"]["pretraining"],"initial_policy_is_untrained":False}
    if args.check_initial:
        with destination.open("x",encoding="utf-8") as f:json.dump(initial_check,f,indent=2)
        print(json.dumps({"initial_check":"passed","result":str(destination)}));return
    if not (baseline.parent/"complete.json").is_file() or not (OUT/"complete.json").is_file():
        raise ValueError("Wait for both registered arms and all their evaluations")
    classical=load_evaluation(Path(plan["classical"]))
    if classical[1]["algorithm"]!="classical":raise ValueError("Expected preserved classical benchmark")
    worlds=plan["evaluation"]["worlds"]
    indices=np.random.default_rng(701).integers(0,worlds,size=(10000,worlds))
    arms={};loaded={};environments={}
    for name,folder in roots.items():
        training=folder/"train";config=configs[name];checked=audit(training)
        summary=read(training/"training_summary.json")
        maximum=config["workers"]*10*config["rollout_steps"]
        if summary["status"]!="complete" or not config["live_steps"]<=summary["live_transitions"]<config["live_steps"]+maximum:
            raise ValueError("Incomplete registered live-transition budget")
        provenance=read(training/"provenance.json")
        environments[name]={k:provenance[k] for k in ("python","packages")}
        records=read(training/"checkpoints.json")
        middle=min((r for r in records if r["file"].startswith("policy-live-") and r["live_transitions"]>=50000),key=lambda r:r["live_transitions"])
        if middle["live_transitions"]>=50000+maximum:raise ValueError("50k checkpoint is outside tolerance")
        selected={"initial":next(r for r in records if r["file"]=="initial-model.zip"),"50k":middle,"final":next(r for r in records if r["file"]=="model.zip")}
        loaded[name]={}
        for stage,record in selected.items():
            data=load_evaluation(folder/f"eval-{stage}-dev20")
            summary_eval,protocol,scenarios,values=data
            if protocol["source_sha256"]!=expected_source(evaluation_source) or protocol["seed"]!=20260 or protocol["episodes"]!=worlds:
                raise ValueError("Evaluation implementation or worlds changed")
            if scenarios!=classical[2] or set(scenarios)!=set(range(worlds)) or summary_eval["agent_episodes"]!=10*worlds:
                raise ValueError("Incomplete or unpaired scenarios")
            if protocol["checkpoint"]!=record or Path(protocol["model_path"]).resolve()!=verified_checkpoint(training,record):
                raise ValueError("Unregistered checkpoint selected")
            for key,default in (("algorithm",None),("guidance",False),("filter",False),("static_filter",False),
                    ("conflict_features",False),("mask_conflict_features",False),("traffic_position_scale",1.),
                    ("action_reference","direct"),("exploration","gaussian"),("initial_action_std",None)):
                if protocol.get(key,default)!=config.get(key,default):raise ValueError("Evaluation changed "+key)
            if protocol["evaluation_progress_scale"]!=0 or protocol["evaluation_reward_scale"]!=1:
                raise ValueError("Native scoring changed")
            if not summary_eval["runtime"]["static_area_filter"] or summary_eval["runtime"]["traffic_conflict_filter"]:
                raise ValueError("Wrong static or traffic action support")
            loaded[name][stage]=data
        with (training/"learning.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
        possible=sum(config["epochs"]*math.ceil(int(row["rollout_live_samples"])/config["batch_size"]) for row in rows)
        previous=0;updates=[]
        for row in rows:
            current=int(row["optimizer_steps"]);updates.append(current-previous);previous=current
        optimizer={"applied_steps":checked["optimizer_steps"],"full_epoch_step_budget":possible,
                   "applied_fraction":checked["optimizer_steps"]/possible,"updates_per_rollout":updates,
                   "one_update_rollout_fraction":float(np.mean(np.array(updates)==1)),
                   "first_rollout_approx_kl":float(rows[0]["approx_kl"]),
                   "mean_logged_approx_kl":float(np.mean([float(r["approx_kl"]) for r in rows])),
                   "kl_note":"Logged approximate KL is SB3's last executed epoch minibatch average, not a complete policy KL bound."}
        means={stage:{m:float(v.mean()) for m,v in loaded[name][stage][3].items()} for stage in STAGES}
        gain={m:paired_effect(loaded[name]["final"][3][m],loaded[name]["initial"][3][m],indices) for m in means["initial"]}
        threshold=plan["advance_screen"]
        checks={"arrival_level":means["final"]["waypoint_reached"]>=threshold["arrival_at_least"],
                "arrival_change":gain["waypoint_reached"]["mean_difference"]>=threshold["arrival_change_at_least"],
                "clean_gain":gain["clean_completion"]["mean_difference"]>=threshold["clean_completion_change_at_least"],
                **{metric:gain[metric]["mean_difference"]<=threshold["safety_time_changes_at_most"] for metric in threshold["metrics"]}}
        curve=read(folder/"curve/curve.json")
        if [p["live_transitions"] for p in curve["points"]]!=[selected[s]["live_transitions"] for s in STAGES]:raise ValueError("Curve omitted a checkpoint")
        for stage,point in zip(STAGES,curve["points"]):
            for metric in means[stage]:
                if abs(point["metrics"][metric]["mean"]-means[stage][metric])>1e-12:raise ValueError("Curve metric mismatch")
        arms[name]={"configuration":config,"audit":checked,"training_summary":summary,"optimizer_diagnostics":optimizer,
                    "means":means,"own_initial_learning":gain,"screen":checks,"passes_screen":all(checks.values()),
                    "checkpoints":selected,"curve":curve,"csv_sha256":{s:loaded[name][s][0]["csv_sha256"] for s in STAGES}}
    if environments["default"]!=environments["smaller"]:raise ValueError("Training dependencies differ")
    for metric in loaded["default"]["initial"][3]:
        if not np.array_equal(loaded["default"]["initial"][3][metric],loaded["smaller"]["initial"][3][metric]):
            raise ValueError("Identical initial policies have different measured behavior")
    contrasts={stage:{m:paired_effect(loaded["smaller"][stage][3][m],loaded["default"][stage][3][m],indices)
                      for m in loaded["default"][stage][3]} for stage in STAGES}
    result={"recorded_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":sha(OUT/"protocol.json"),
            "analyzer_sha256":sha(Path(__file__)),"both_registered_arms_included":True,"initial_check":initial_check,
            "recovery":plan["recovery"],"recovery_prefix_check":read(OUT/"recovery-prefix-check.json"),
            "training_seeds":1,"worlds":worlds,"arms":arms,"smaller_minus_default":contrasts,
            "classical_means":{k:float(v.mean()) for k,v in classical[3].items()},
            "limitations":plan["limitations"]+["World bootstrap intervals do not measure training-seed uncertainty."],
            "candidate_promoted":False,"unseen_scenarios_used":False}
    with destination.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"result":str(destination),"screen":{k:v["passes_screen"] for k,v in arms.items()}}))

if __name__=="__main__":main()
