"""Analyze both registered gSDE timing arms without selecting a winning checkpoint."""
from pathlib import Path
import csv,hashlib,io,json,math,statistics,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/"runs/cluster-ppo-gsde-v1"
PROTOCOL=ROOT/"jobs/ppo_gsde_v1.json"
sys.path.insert(0,str(ROOT))
from scripts.analyze_rl_cohort import paired_effect,screen
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))

def analyze_seed(plan,seed):
    import torch
    sys.path.insert(0,plan["training_source"])
    from atc_rl.cluster import verify
    from atc_rl.audit import audit
    from atc_rl.compare import load_evaluation
    from atc_rl.checkpoint_identity import verified_checkpoint
    source=Path(plan["training_source"])
    verify(source)
    manifest=read(source/"cluster-manifest.json")["files"]
    packages=("atc","atc_rl","core","bluesky_gym","bluesky_zoo")
    evaluation_source={k:v for k,v in manifest.items() if k.endswith(".py") and k.split("/")[0] in packages}
    expected_source={**evaluation_source,"pyproject.toml":manifest["pyproject.toml"]}

    worlds=plan["evaluation_worlds"]
    indices=np.random.default_rng(701).integers(0,worlds,size=(10000,worlds))
    classical=load_evaluation(PARENT/f"seed-{seed}"/"every1/evaluations/classical")
    arms={};canonical_environment=None
    for arm in ("every1","every12"):
        parent=PARENT/f"seed-{seed}"/arm
        training=parent/"train"
        checked=audit(training)
        maximum=plan["config"]["workers"]*10*plan["config"]["rollout_steps"]
        if checked["status"]!="complete" or not plan["config"]["live_steps"]<=checked["live_transitions"]<plan["config"]["live_steps"]+maximum:
            raise ValueError("Incomplete registered training")
        config=read(training/"config.json")
        if config['world_seeds']!=[seed+10*i for i in range(plan['config']['workers'])]:
            raise ValueError('Simulator training seeds differ')
        expected={**plan["config"],**plan["arms"][arm],"seed":seed}
        if any(config.get(k)!=v for k,v in expected.items()):raise ValueError("Training changed the registered recipe")
        if read(parent/"launch.json")["protocol_sha256"]!=hashlib.sha256((PROTOCOL).read_bytes()).hexdigest():
            raise ValueError("Protocol changed after launch")
        provenance=read(training/"provenance.json")
        if provenance["source_sha256"]!=expected_source:raise ValueError("Training source changed")
        environment={k:provenance[k] for k in ("python","packages")}
        if canonical_environment is None:canonical_environment=environment
        if environment!=canonical_environment:raise ValueError("Training dependencies differ")
        runtime=read(training/"runtime.json")
        if not runtime["static_area_filter"] or runtime["traffic_conflict_filter"]:raise ValueError("Training action support changed")
        assert config["initial_action_std"] is None and config["gae_lambda"]==.95
        assert config["noise_resampling_seconds"]==5*expected["sde_sample_freq"]
        records=read(training/"checkpoints.json")
        from atc_rl.matrix_evaluate import select_checkpoint
        selected={stage:select_checkpoint(training,stage)[1] for stage in plan['stages']}
        evaluated={}
        for stage in plan["stages"]:
            loaded=load_evaluation(parent/"evaluations"/stage)
            summary,protocol,scenarios,values=loaded
            if protocol["source_sha256"]!=evaluation_source or protocol["seed"]!=plan["evaluation_seed"] or protocol["episodes"]!=worlds:
                raise ValueError("Evaluation source or scenarios changed")
            if set(scenarios)!=set(range(worlds)) or summary["agent_episodes"]!=10*worlds:
                raise ValueError("Incomplete evaluation")
            if scenarios!=classical[2]:raise ValueError("Unpaired classical scenarios")
            for key,default in (("exploration","gaussian"),("sde_weight_std",None),("sde_sample_freq",None),("algorithm",None),("guidance",False),("filter",False),("static_filter",False),
                                ("conflict_features",False),("mask_conflict_features",False),("action_reference","direct"),
                                ("initial_action_std",None),("neutral_action_mean",False)):
                if protocol.get(key,default)!=config.get(key,default):raise ValueError("Evaluation changed "+key)
            if protocol["evaluation_progress_scale"]!=0 or protocol["evaluation_reward_scale"]!=1:
                raise ValueError("Native evaluation rewards changed")
            if not summary["runtime"]["static_area_filter"] or summary["runtime"]["traffic_conflict_filter"]:
                raise ValueError("Evaluation action support changed")
            if protocol["checkpoint"]!=selected[stage]:raise ValueError("Wrong evaluated checkpoint")
            verified_checkpoint(training,selected[stage])
            evaluated[stage]=loaded
        if selected["initial"]["live_transitions"]!=0 or selected["initial"]["optimizer_steps"]!=0:
            raise ValueError("Initial policy was trained")
        if selected["final"]["sha256"]!=checked["model_sha256"]:raise ValueError("Final policy differs from audit")
        curve=read(parent/"curve/curve.json")
        own_classical=load_evaluation(parent/'evaluations/classical')
        if own_classical[2]!=classical[2] or any(not np.array_equal(v,classical[3][m]) for m,v in own_classical[3].items()):
            raise ValueError('Repeated fixed classical evaluations differ')
        if curve['classical_means']!={m:float(v.mean()) for m,v in classical[3].items()}:
            raise ValueError('Curve classical reference differs')
        from atc_rl.checkpoint_identity import policy_fingerprint
        for point,stage in zip(curve['points'],plan['stages']):
            if point['policy_fingerprint']!=policy_fingerprint(training/selected[stage]['file']):
                raise ValueError('Curve checkpoint identity differs')
        if [p["live_transitions"] for p in curve["points"]]!=[selected[s]["live_transitions"] for s in plan["stages"]]:
            raise ValueError("Learning curve omitted a registered checkpoint")
        with zipfile.ZipFile(training/"initial-model.zip") as archive:
            state=torch.load(io.BytesIO(archive.read("policy.pth")),map_location="cpu",weights_only=True)
        if not torch.allclose(state["log_std"].exp(),torch.full_like(state["log_std"],expected["sde_weight_std"])):
            raise ValueError("Initial exploration does not match the recipe")
        with (training/"learning.csv").open(newline="",encoding="utf-8") as stream:
            learning_rows=list(csv.DictReader(stream))
        planned=sum(config["epochs"]*math.ceil(int(row["rollout_live_samples"])/config["batch_size"]) for row in learning_rows)
        if int(learning_rows[-1]["optimizer_steps"])!=checked["optimizer_steps"]:
            raise ValueError("Optimizer accounting differs")
        optimizer={"applied_steps":checked["optimizer_steps"],"full_budget_steps":planned,
                   "applied_fraction":checked["optimizer_steps"]/planned}
        arms[arm]={"config":config,"audit":checked,"data":evaluated,"state":state,"curve":curve,"optimizer":optimizer}
    left,right=arms["every1"],arms["every12"]
    ignore={"run_dir","sde_sample_freq","noise_resampling_seconds"}
    if {k:v for k,v in left["config"].items() if k not in ignore}!={k:v for k,v in right["config"].items() if k not in ignore}:
        raise ValueError("Noise-timing comparison changed another setting")
    if left["state"].keys()!=right["state"].keys() or [k for k in left["state"] if not torch.equal(left["state"][k],right["state"][k])]!=[]:
        raise ValueError("Initial policy tensors differ")
    effects={};arm_results={}
    for metric in left["data"]["initial"][3]:
        if not np.array_equal(left["data"]["initial"][3][metric],right["data"]["initial"][3][metric]):
            raise ValueError("Deterministic initial behavior differs")
        gain={arm:data["data"]["final"][3][metric].astype(float)-data["data"]["initial"][3][metric].astype(float) for arm,data in arms.items()}
        effects[metric]=paired_effect(gain["every12"],gain["every1"],indices)
    for arm,data in arms.items():
        means={stage:{metric:float(v.mean()) for metric,v in loaded[3].items()} for stage,loaded in data["data"].items()}
        checks=screen(means["initial"],means["final"],plan["advance_screen"])
        arm_results[arm]={"audit":data["audit"],"means":means,"screen_checks":checks,"screen_passed":all(checks.values()),"optimizer":data["optimizer"],
          "learning_effects":{metric:paired_effect(v,data["data"]["initial"][3][metric],indices) for metric,v in data["data"]["final"][3].items()},
          "trained_minus_classical":{metric:paired_effect(v,classical[3][metric],indices) for metric,v in data["data"]["final"][3].items()},
          "curve":data["curve"],"csv_sha256":{s:d[0]["csv_sha256"] for s,d in data["data"].items()}}
    result={"seed":seed,"both_registered_arms_included":True,"arms":arm_results,"persistent_minus_frequent_learning":effects,
            "full_initial_policy_tensors_identical":True,"initial_physical_behavior_identical":True,
            "training_environment":canonical_environment,
            "protocol_sha256":hashlib.sha256((PROTOCOL).read_bytes()).hexdigest(),
            "unseen_scenarios_used":False,"limitations":plan["limitations"]}
    return result


def mean_sd(values):
    values=[float(v) for v in values]
    return {"mean":statistics.mean(values),"sample_standard_deviation":statistics.stdev(values),
            "per_seed_values":values,"minimum":min(values),"maximum":max(values)}


def main():
    plan=read(PROTOCOL)
    destination=PARENT/"three-seed-results.json"
    if destination.exists():raise FileExistsError(destination)
    from atc_rl.cluster import verify
    verify(ROOT)
    if plan['seeds']!=[52600,52700,52800]:raise ValueError('Registered fresh seeds changed')
    plan['training_source']=str(ROOT)
    plan['evaluation_seed']=plan['evaluation']['seed']
    plan['evaluation_worlds']=plan['evaluation']['worlds']
    plan['stages']=plan['evaluation']['stages']
    protocol_sha=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    for row in plan['rows']:
        parent=PARENT/f"seed-{row['seed']}"/row['arm']
        for stage in ('train','evaluate'):
            marker=read(parent/f'{stage}-complete.json')
            if marker['protocol_sha256']!=protocol_sha or marker['completed']!=stage or marker['seed']!=row['seed'] or marker['arm']!=row['arm']:
                raise ValueError('Incomplete or changed registered row')
        reload=read(parent/'cuda-reload.json')
        if not reload['bounded_finite_action'] or reload['device']!='cuda':raise ValueError('Missing CUDA reload check')
        summary=read(parent/'train/training_summary.json')
        if reload['model_sha256']!=summary['model_sha256'] or reload['sde_sample_freq']!=plan['arms'][row['arm']]['sde_sample_freq']:
            raise ValueError('CUDA reload used another policy')
    first_row=plan['rows'][0]
    native=read(PARENT/f"seed-{first_row['seed']}"/first_row['arm']/'native-check.json')
    if native['matched_metric_values']!=180:raise ValueError('Native harness parity did not pass')
    pairs={seed:analyze_seed(plan,seed) for seed in plan["seeds"]}
    first=pairs[plan["seeds"][0]]
    if any(pair["training_environment"]!=first["training_environment"] for pair in pairs.values()):
        raise ValueError("Dependencies differ across training seeds")
    metrics=list(first["persistent_minus_frequent_learning"])
    aggregate={}
    for arm in ("every1","every12"):
        means={stage:{metric:statistics.mean(pair["arms"][arm]["means"][stage][metric] for pair in pairs.values()) for metric in metrics} for stage in plan["stages"]}
        checks=screen(means["initial"],means["final"],plan["advance_screen"])
        aggregate[arm]={"mean_physical_metrics":means,"screen_checks_on_fresh_seed_mean":checks,
            "screen_passed_on_fresh_seed_mean":all(checks.values()),
            "screen_passed_by_seed":{seed:pair["arms"][arm]["screen_passed"] for seed,pair in pairs.items()},
            "own_initial_learning":{metric:mean_sd(pair["arms"][arm]["learning_effects"][metric]["mean_difference"] for pair in pairs.values()) for metric in metrics},
            "optimizer":{seed:pair["arms"][arm]["optimizer"] for seed,pair in pairs.items()}}
    contrasts={metric:mean_sd(pair["persistent_minus_frequent_learning"][metric]["mean_difference"] for pair in pairs.values()) for metric in metrics}
    result={"primary_seeds":plan["seeds"],"all_six_registered_runs_included":True,"selected_pilot_excluded_from_primary_aggregate":True,
            "selected_pilot_for_context":plan["selected_pilot"],"fresh_seed_aggregate":aggregate,
            "persistent_minus_frequent_learning":contrasts,"per_seed":pairs,
            "protocol_sha256":hashlib.sha256((PROTOCOL).read_bytes()).hexdigest(),
            "uncertainty":"Sample SD describes variability across the three fresh training seeds. Per-seed bootstrap intervals resample the same twenty development worlds and do not establish unseen-scenario generalization.",
            "unseen_scenarios_used":False,"limitations":plan["limitations"]}
    with destination.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({"results":str(destination),"persistent_minus_frequent_learning":contrasts}))
if __name__=="__main__":main()
