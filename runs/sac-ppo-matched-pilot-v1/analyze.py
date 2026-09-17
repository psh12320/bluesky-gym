"""Audit the registered SAC/PPO recipes and their paired learning contributions."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.analyze_rl_cohort import paired_effect,screen
read=lambda path:json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    import torch
    plan=read(PARENT/"protocol.json")
    destination=PARENT/"paired-results.json"
    if destination.exists():raise FileExistsError(destination)
    if not (PARENT/"complete.json").is_file():raise ValueError("Both registered arms must complete first")
    source=Path(plan["source"])
    sys.path.insert(0,str(source))
    from atc_rl.cluster import verify
    from atc_rl.checkpoint_identity import verified_checkpoint,policy_fingerprint
    from atc_rl.compare import load_evaluation
    verify(source)
    if hashlib.sha256((ROOT/"runs/sac-comparison-source-v1.zip").read_bytes()).hexdigest()!=plan["source_bundle_sha256"]:
        raise ValueError("Source archive differs from the registered plan")
    manifest=read(source/"cluster-manifest.json")["files"]
    packages=("atc","atc_rl","core","bluesky_gym","bluesky_zoo")
    evaluation_source={k:v for k,v in manifest.items() if k.endswith(".py") and k.split("/")[0] in packages}
    expected_source={**evaluation_source,"pyproject.toml":manifest["pyproject.toml"]}
    seed=plan["seeds"][0];worlds=plan["evaluation_worlds"]
    if plan["seeds"]!=[seed]:raise ValueError("This analyzer handles the registered one-seed pilot")
    indices=np.random.default_rng(701).integers(0,worlds,size=(10000,worlds))
    classical=load_evaluation(Path(plan["classical_evaluation"]))
    arms={};canonical_environment=None
    for arm in ("sac","ppo"):
        parent=PARENT/f"seed-{seed}"/arm;training=parent/"train"
        audit_path=parent/"analysis-checkpoint-audit.json"
        result=subprocess.run([sys.executable,"-m",plan["modules"][arm]["audit"],"--run",str(training),"--out",str(audit_path)],
                              cwd=source,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"},capture_output=True,text=True)
        if result.returncode:raise RuntimeError("Checkpoint audit failed: "+result.stdout+result.stderr)
        checked=read(audit_path)
        maximum=plan["collection_slots"][arm];budget=plan["common"]["live_steps"]
        if checked["status"]!="complete" or not budget<=checked["live_transitions"]<budget+maximum:
            raise ValueError("Incomplete registered training")
        config=read(training/"config.json")
        expected={**plan["common"],**plan["arms"][arm],"algorithm":arm,"seed":seed}
        if any(config.get(k)!=v for k,v in expected.items()):raise ValueError("Training changed the registered recipe")
        launch=read(parent/"launch.json")
        if launch["protocol_sha256"]!=hashlib.sha256((PARENT/"protocol.json").read_bytes()).hexdigest():
            raise ValueError("Protocol changed after launch")
        provenance=read(training/"provenance.json")
        if provenance["source_sha256"]!=expected_source:raise ValueError("Training source changed")
        environment={k:provenance[k] for k in ("python","packages")}
        if canonical_environment is None:canonical_environment=environment
        if environment!=canonical_environment:raise ValueError("Training dependencies differ")
        runtime=read(training/"runtime.json")
        if not runtime["static_area_filter"] or runtime["traffic_conflict_filter"]:raise ValueError("Training action support changed")
        if runtime["traffic_position_scale"]!=1. or runtime["decision_interval_seconds"]!=5 or runtime["episode_time_limit"]!=3000:
            raise ValueError("Training input/time settings changed")
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
                raise ValueError("Evaluation implementation or scenario count changed")
            if set(scenarios)!=set(range(worlds)) or summary["agent_episodes"]!=10*worlds or scenarios!=classical[2]:
                raise ValueError("Incomplete or unpaired evaluation")
            for key,default in (("algorithm",None),("guidance",False),("filter",False),("static_filter",False),
                ("action_reference","direct"),("conflict_features",False),("mask_conflict_features",False),
                ("traffic_position_scale",1.),("initial_action_std",None),("neutral_action_mean",False),
                ("exploration","gaussian"),("sde_weight_std",None),("sde_sample_freq",None)):
                if protocol.get(key,default)!=config.get(key,default):raise ValueError("Evaluation changed "+key)
            if protocol["evaluation_reward_scale"]!=1 or protocol["evaluation_progress_scale"]!=0:
                raise ValueError("Evaluation reward changed")
            if not summary["runtime"]["static_area_filter"] or summary["runtime"]["traffic_conflict_filter"]:
                raise ValueError("Evaluation action support changed")
            if summary["runtime"]["traffic_position_scale"]!=1.:raise ValueError("Evaluation input scaling changed")
            if protocol["checkpoint"]!=selected[stage]:raise ValueError("Wrong evaluated checkpoint")
            path=verified_checkpoint(training,selected[stage])
            if Path(protocol["model_path"]).resolve()!=path:raise ValueError("Evaluation model path differs from its lineage")
            evaluated[stage]=loaded
        if selected["initial"]["live_transitions"] or selected["initial"]["optimizer_steps"]:raise ValueError("Initial model was trained")
        if selected["final"]["sha256"]!=checked["model_sha256"]:raise ValueError("Audited final checkpoint differs")
        curve=read(parent/"curve/curve.json")
        if [p["live_transitions"] for p in curve["points"]]!=[selected[s]["live_transitions"] for s in plan["stages"]]:
            raise ValueError("Learning curve omitted a registered checkpoint")
        for point,stage in zip(curve["points"],plan["stages"]):
            if point["policy_fingerprint"]!=policy_fingerprint(training/selected[stage]["file"]):
                raise ValueError("Learning curve checkpoint identity differs")
        with zipfile.ZipFile(training/"initial-model.zip") as archive:
            state=torch.load(io.BytesIO(archive.read("policy.pth")),map_location="cpu",weights_only=True)
        prefix="actor.mu" if arm=="sac" else "action_net"
        if torch.count_nonzero(state[prefix+".weight"]) or torch.count_nonzero(state[prefix+".bias"]):
            raise ValueError("Initial actor mean is not zero")
        log_std=state["actor.log_std.bias"] if arm=="sac" else state["log_std"]
        if not torch.allclose(log_std.exp(),torch.full_like(log_std,.05)):raise ValueError("Initial latent std differs")
        if arm=="sac" and torch.count_nonzero(state["actor.log_std.weight"]):raise ValueError("Initial SAC noise is not state-independent")
        arms[arm]={"config":config,"audit":checked,"data":evaluated,"curve":curve,
                   "training_summary":read(training/"training_summary.json")}
    for key in (*plan["common"].keys(),"gamma","progress_scale","neutral_action_mean"):
        if arms["sac"]["config"].get(key)!=arms["ppo"]["config"].get(key):raise ValueError("Shared setting differs: "+key)
    effects={};arm_results={}
    for metric in arms["sac"]["data"]["initial"][3]:
        if not np.array_equal(arms["sac"]["data"]["initial"][3][metric],arms["ppo"]["data"]["initial"][3][metric]):
            raise ValueError("Initial deterministic physical behavior differs")
        gain={arm:data["data"]["final"][3][metric].astype(float)-data["data"]["initial"][3][metric].astype(float) for arm,data in arms.items()}
        effects[metric]=paired_effect(gain["sac"],gain["ppo"],indices)
    for arm,data in arms.items():
        means={stage:{metric:float(v.mean()) for metric,v in loaded[3].items()} for stage,loaded in data["data"].items()}
        checks=screen(means["initial"],means["final"],plan["advance_screen"])
        arm_results[arm]={"audit":data["audit"],"training_summary":data["training_summary"],"means":means,"screen_checks":checks,"screen_passed":all(checks.values()),
            "learning_effects":{metric:paired_effect(v,data["data"]["initial"][3][metric],indices) for metric,v in data["data"]["final"][3].items()},
            "trained_minus_classical":{metric:paired_effect(v,classical[3][metric],indices) for metric,v in data["data"]["final"][3].items()},
            "curve":data["curve"],"csv_sha256":{stage:loaded[0]["csv_sha256"] for stage,loaded in data["data"].items()}}
    result={"seed":seed,"both_registered_arms_included":True,"initial_physical_behavior_identical":True,
            "network_tensors_matched":False,"single_mechanism_isolated":False,"arms":arm_results,
            "sac_minus_ppo_learning":effects,"training_environment":canonical_environment,
            "protocol_sha256":hashlib.sha256((PARENT/"protocol.json").read_bytes()).hexdigest(),
            "unseen_scenarios_used":False,"limitations":plan["limitations"]}
    with destination.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({"result":str(destination),"sac_minus_ppo_learning":effects}))
if __name__=="__main__":main()
