"""Analyze the isolated GAE comparison with the unchanged, separately registered PPO control."""
from pathlib import Path
import hashlib,io,json,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.analyze_rl_cohort import paired_effect,screen
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))

def main():
    import torch
    plan=read(PARENT/"protocol.json")
    destination=PARENT/"paired-results.json"
    if destination.exists():raise FileExistsError(destination)
    sys.path.insert(0,plan["source"])
    from atc_rl.cluster import verify
    from atc_rl.audit import audit
    from atc_rl.compare import load_evaluation
    from atc_rl.checkpoint_identity import verified_checkpoint
    source=Path(plan["source"])
    verify(source)
    manifest=read(source/"cluster-manifest.json")["files"]
    packages=("atc","atc_rl","core","bluesky_gym","bluesky_zoo")
    evaluation_source={k:v for k,v in manifest.items() if k.endswith(".py") and k.split("/")[0] in packages}
    expected_source={**evaluation_source,"pyproject.toml":manifest["pyproject.toml"]}
    if not (PARENT/"lambda099-complete.json").is_file():raise ValueError("The new lambda arm is incomplete")
    control_plan=Path(plan["control_protocol"])
    if not (control_plan.parent/"complete.json").is_file():raise ValueError("The separately registered control study is incomplete")
    if hashlib.sha256(control_plan.read_bytes()).hexdigest()!=plan["control_protocol_sha256"]:
        raise ValueError("Control protocol changed")
    if hashlib.sha256((ROOT/"runs/sac-comparison-source-v1.zip").read_bytes()).hexdigest()!=plan["source_bundle_sha256"]:
        raise ValueError("Frozen training archive changed")
    seed=plan["seed"]
    worlds=plan["evaluation_worlds"]
    indices=np.random.default_rng(701).integers(0,worlds,size=(10000,worlds))
    classical=load_evaluation(Path(plan["classical_evaluation"]))
    arms={};canonical_environment=None
    for arm in ("lambda095","lambda099"):
        parent=Path(plan["control_directory"] if arm=="lambda095" else plan["new_directory"])
        training=parent/"train"
        checked=audit(training)
        maximum=plan["config"]["workers"]*10*plan["config"]["rollout_steps"]
        if checked["status"]!="complete" or not plan["config"]["live_steps"]<=checked["live_transitions"]<plan["config"]["live_steps"]+maximum:
            raise ValueError("Incomplete registered training")
        config=read(training/"config.json")
        expected={**plan["config"],**plan["arms"][arm],"seed":seed}
        if any(config.get(k)!=v for k,v in expected.items()):raise ValueError("Training changed the registered recipe")
        if checked["gae_lambda"]!=expected["gae_lambda"] or checked["gamma"]!=config["gamma"]:
            raise ValueError("Loaded critic/rollout credit settings differ")
        launch=read(parent/"launch.json")
        expected_protocol=plan["control_protocol_sha256"] if arm=="lambda095" else hashlib.sha256((PARENT/"protocol.json").read_bytes()).hexdigest()
        if launch["protocol_sha256"]!=expected_protocol:raise ValueError("Launch protocol identity differs")
        provenance=read(training/"provenance.json")
        if provenance["source_sha256"]!=expected_source:raise ValueError("Training source changed")
        environment={k:provenance[k] for k in ("python","packages")}
        if canonical_environment is None:canonical_environment=environment
        if environment!=canonical_environment:raise ValueError("Training dependencies differ")
        runtime=read(training/"runtime.json")
        if not runtime["static_area_filter"] or runtime["traffic_conflict_filter"]:raise ValueError("Training action support changed")
        if runtime["traffic_position_scale"]!=expected["traffic_position_scale"]:raise ValueError("Runtime scaling differs")
        records=read(training/"checkpoints.json")
        middle=min((r for r in records if r["file"].startswith("policy-live-") and r["live_transitions"]>=50000),key=lambda r:r["live_transitions"])
        if middle["live_transitions"]>=50000+maximum:raise ValueError("Wrong intermediate budget")
        selected={"initial":next(r for r in records if r["file"]=="initial-model.zip"),"50k":middle,
                  "final":next(r for r in records if r["file"]=="model.zip")}
        evaluated={}
        for stage in plan["stages"]:
            loaded=load_evaluation(parent/f"eval-{stage}-dev20")
            summary,protocol,scenarios,values=loaded
            if protocol["source_sha256"]!=evaluation_source or protocol["seed"]!=plan["evaluation_seed"] or protocol["episodes"]!=worlds:
                raise ValueError("Evaluation source or scenarios changed")
            if set(scenarios)!=set(range(worlds)) or summary["agent_episodes"]!=10*worlds:
                raise ValueError("Incomplete evaluation")
            if scenarios!=classical[2]:raise ValueError("Unpaired classical scenarios")
            for key,default in (("traffic_position_scale",1.0),("exploration","gaussian"),("sde_weight_std",None),("sde_sample_freq",None),("algorithm",None),("guidance",False),("filter",False),("static_filter",False),
                                ("conflict_features",False),("mask_conflict_features",False),("action_reference","direct"),
                                ("initial_action_std",None),("neutral_action_mean",False)):
                if protocol.get(key,default)!=config.get(key,default):raise ValueError("Evaluation changed "+key)
            if protocol["evaluation_progress_scale"]!=0 or protocol["evaluation_reward_scale"]!=1:
                raise ValueError("Native evaluation rewards changed")
            if not summary["runtime"]["static_area_filter"] or summary["runtime"]["traffic_conflict_filter"]:
                raise ValueError("Evaluation action support changed")
            if summary["runtime"]["traffic_position_scale"]!=config["traffic_position_scale"]:raise ValueError("Evaluation runtime scaling differs")
            if protocol["checkpoint"]!=selected[stage]:raise ValueError("Wrong evaluated checkpoint")
            verified_checkpoint(training,selected[stage])
            evaluated[stage]=loaded
        if selected["initial"]["live_transitions"]!=0 or selected["initial"]["optimizer_steps"]!=0:
            raise ValueError("Initial policy was trained")
        if selected["final"]["sha256"]!=checked["model_sha256"]:raise ValueError("Final policy differs from audit")
        curve=read(parent/"curve/curve.json")
        if [p["live_transitions"] for p in curve["points"]]!=[selected[s]["live_transitions"] for s in plan["stages"]]:
            raise ValueError("Learning curve omitted a registered checkpoint")
        with zipfile.ZipFile(training/"initial-model.zip") as archive:
            state=torch.load(io.BytesIO(archive.read("policy.pth")),map_location="cpu",weights_only=True)
        if not torch.allclose(state["log_std"].exp(),torch.full_like(state["log_std"],expected["initial_action_std"])):
            raise ValueError("Initial exploration does not match the recipe")
        arms[arm]={"config":config,"audit":checked,"data":evaluated,"state":state,"curve":curve}
    left,right=arms["lambda095"],arms["lambda099"]
    ignore={"run_dir","gae_lambda"}
    if {k:v for k,v in left["config"].items() if k not in ignore}!={k:v for k,v in right["config"].items() if k not in ignore}:
        raise ValueError("Credit-assignment comparison changed another setting")
    if left["state"].keys()!=right["state"].keys() or [k for k in left["state"] if not torch.equal(left["state"][k],right["state"][k])]!=[]:
        raise ValueError("Initial tensors differ")
    effects={};arm_results={}
    for metric in left["data"]["initial"][3]:
        if not np.array_equal(left["data"]["initial"][3][metric],right["data"]["initial"][3][metric]):
            raise ValueError("Deterministic initial behavior differs")
        gain={arm:data["data"]["final"][3][metric].astype(float)-data["data"]["initial"][3][metric].astype(float) for arm,data in arms.items()}
        effects[metric]=paired_effect(gain["lambda099"],gain["lambda095"],indices)
    for arm,data in arms.items():
        means={stage:{metric:float(v.mean()) for metric,v in loaded[3].items()} for stage,loaded in data["data"].items()}
        checks=screen(means["initial"],means["final"],plan["advance_screen"])
        arm_results[arm]={"audit":data["audit"],"means":means,"screen_checks":checks,"screen_passed":all(checks.values()),
          "learning_effects":{metric:paired_effect(v,data["data"]["initial"][3][metric],indices) for metric,v in data["data"]["final"][3].items()},
          "trained_minus_classical":{metric:paired_effect(v,classical[3][metric],indices) for metric,v in data["data"]["final"][3].items()},
          "curve":data["curve"],"csv_sha256":{s:d[0]["csv_sha256"] for s,d in data["data"].items()}}
    result={"seed":seed,"both_registered_arms_included":True,"arms":arm_results,"higher_minus_default_learning":effects,
            "full_initial_policy_tensors_identical":True,"initial_physical_behavior_identical":True,
            "control_reused":True,"control_directory":plan["control_directory"],
            "control_protocol_sha256":plan["control_protocol_sha256"],"only_training_setting_changed":"gae_lambda",
            "training_environment":canonical_environment,
            "protocol_sha256":hashlib.sha256((PARENT/"protocol.json").read_bytes()).hexdigest(),
            "unseen_scenarios_used":False,"limitations":plan["limitations"]}
    with destination.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"results":str(destination),"higher_minus_default_learning":effects}))
if __name__=="__main__":main()
