"""Summarize all seeds of one completed registered support group."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.analyze_rl_cohort import paired_effect,aggregate_effects
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group",choices=["g0-f1","g1-f1"])
    args=parser.parse_args()
    plan=read(PARENT/"protocol.json")
    group=next(g for g in plan["support_groups"] if g["name"]==args.group)
    out=PARENT/args.group/"completed-three-seed-results.json"
    if out.exists():raise FileExistsError(out)
    sys.path.insert(0,plan["training_source"])
    from atc_rl.cluster import verify
    from atc_rl.audit import audit
    from atc_rl.algorithm_compare import initial_for
    from atc_rl.checkpoint_identity import verified_checkpoint
    from atc_rl.compare import load_evaluation
    source=Path(plan["training_source"]);evaluation=Path(plan["evaluation_source"])
    verify(source);verify(evaluation)
    package_names=("atc","atc_rl","core","bluesky_gym","bluesky_zoo")
    def sources(path):
        return {k:v for k,v in read(path/"cluster-manifest.json")["files"].items()
                if k.endswith(".py") and k.split("/")[0] in package_names}
    expected_training={**sources(source),"pyproject.toml":hashlib.sha256((source/"pyproject.toml").read_bytes()).hexdigest()}
    expected_evaluation=sources(evaluation)
    classical=load_evaluation(Path(plan["classical_evaluation"]))
    worlds=plan["evaluation_worlds"]
    indices=np.random.default_rng(701).integers(0,worlds,size=(10000,worlds))
    rows=[];environment=None
    for seed in plan["seeds"]:
        parent=PARENT/args.group/f"seed-{seed}";training=parent/"train"
        config=read(training/"config.json");checked=audit(training)
        expected={**plan["config"],"seed":seed,"guidance":group["guidance"],"filter":group["filter"]}
        if any(config.get(k)!=v for k,v in expected.items()):raise ValueError("Recipe changed")
        if config["world_seeds"]!=plan["world_seeds"][str(seed)]:raise ValueError("Training world seeds changed")
        if any(config.get(k,False) for k in ("static_filter","conflict_features","mask_conflict_features")):
            raise ValueError("Separate experiment mixed into the support comparison")
        if checked["status"]!="complete" or not plan["config"]["live_steps"]<=checked["live_transitions"]<plan["config"]["live_steps"]+5120:
            raise ValueError("Incomplete or wrong training budget")
        provenance=read(training/"provenance.json")
        if provenance["source_sha256"]!=expected_training:raise ValueError("Source differs from the registered snapshot")
        observed_environment={k:provenance[k] for k in ("python","packages")}
        if environment is None:environment=observed_environment
        if observed_environment!=environment:raise ValueError("Training dependencies differ")
        loaded={stage:load_evaluation(parent/f"eval-{stage}-dev20") for stage in ("initial","final")}
        for stage,(summary,protocol,scenarios,values) in loaded.items():
            if summary["episodes"]!=worlds or summary["agent_episodes"]!=10*worlds or set(scenarios)!=set(range(worlds)):
                raise ValueError("Incomplete evaluation")
            if scenarios!=classical[2]:raise ValueError("Unpaired scenarios")
            if protocol["seed"]!=plan["evaluation_seed"] or protocol["source_sha256"]!=expected_evaluation:
                raise ValueError("Wrong evaluation stream or source")
            for key,default in (("algorithm",None),("guidance",False),("filter",False),("static_filter",False),
                                ("action_reference","direct"),("conflict_features",False),("mask_conflict_features",False)):
                if protocol.get(key,default)!=config.get(key,default):raise ValueError("Evaluation support changed")
            if protocol["evaluation_reward_scale"]!=1 or protocol["evaluation_progress_scale"]!=0:
                raise ValueError("Evaluation rewards changed")
            verified_checkpoint(training,protocol["checkpoint"])
        final=verified_checkpoint(training,loaded["final"][1]["checkpoint"])
        if loaded["final"][1]["checkpoint"]["sha256"]!=checked["model_sha256"]:raise ValueError("Wrong final checkpoint")
        initial_for(final,loaded["initial"][1])
        means={stage:{k:float(v.mean()) for k,v in data[3].items()} for stage,data in loaded.items()}
        individual=read(parent/"comparison/comparison.json")
        for stage,label in (("initial","initial"),("final","trained")):
            if any(abs(value-individual["means"][label][key])>1e-12 for key,value in means[stage].items()):
                raise ValueError("Original comparison disagrees with physical CSV")
        learning={metric:paired_effect(values,loaded["initial"][3][metric],indices) for metric,values in loaded["final"][3].items()}
        rows.append({"seed":seed,"audit":checked,"means":means,"learning_effects":learning,
                     "csv_sha256":{stage:data[0]["csv_sha256"] for stage,data in loaded.items()}})
    result={"group":args.group,"training_seeds":plan["seeds"],"all_three_registered_seeds_included":True,
            "rows":rows,"learning_changes":aggregate_effects(rows,plan["seeds"]),
            "training_environment":environment,
            "protocol_sha256":hashlib.sha256((PARENT/"protocol.json").read_bytes()).hexdigest(),
            "advancement_policy":plan["advancement_policy"],
            "limitations":plan["limitations"]+["This summarizes one completed support group. The separately registered three-cell support analysis remains pending until every group finishes."]}
    with out.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"result":str(out),"learning_changes":result["learning_changes"]}))
if __name__=="__main__":main()
