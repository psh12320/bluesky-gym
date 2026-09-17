"""Analyze completed filter cohorts and their matching guidance-only controls."""
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
from atc_rl.algorithm_compare import initial_for
from atc_rl.checkpoint_identity import verified_checkpoint
from atc_rl.cluster import verify

def main():
    destination=PARENT/"three-seed-support-results.json"
    if destination.exists():raise FileExistsError(destination)
    verify(Path(plan["training_source"]));verify(Path(plan["evaluation_source"]))
    guide=read(ROOT/"runs/ppo-guidance-replication-v1/three-seed-results.json")
    if guide["training_seeds"]!=plan["seeds"]:raise ValueError("Training seeds differ")
    if not guide["all_registered_seeds_included"] or not guide["source_and_recipes_identical_except_seeds"]:
        raise ValueError("Guidance-only cohort is not verified")
    classical=load_evaluation(Path(plan["classical_evaluation"]))
    rows=[]
    specifications=[]
    for group in plan["support_groups"]:
        for seed in plan["seeds"]:
            folder=PARENT/group["name"]/f"seed-{seed}"
            specifications.append((group["name"],seed,folder/"train",folder/"eval-initial-dev20",folder/"eval-final-dev20",group))
    for seed in plan["seeds"]:
        folder=ROOT/("runs/ppo-guidance-pilot-v1" if seed==49900 else f"runs/ppo-guidance-replication-v1/seed-{seed}")
        initial=ROOT/"runs/initial-support-ablation-v1/eval-g1-f0-dev20" if seed==49900 else folder/"eval-initial-dev20"
        specifications.append(("g1-f0",seed,folder/"train",initial,folder/"eval-final-dev20",{"guidance":True,"filter":False}))
    canonical_source=None;canonical_evaluation=None
    for group,seed,training,initial_dir,final_dir,support in specifications:
        checked=audit(training);config=read(training/"config.json")
        if checked["status"]!="complete":raise ValueError("Incomplete training retained: "+str(training))
        for key,value in plan["config"].items():
            if config[key]!=value:raise ValueError("Recipe changed: "+key)
        if config["seed"]!=seed or config["world_seeds"]!=plan["world_seeds"][str(seed)]:raise ValueError("Wrong seed")
        for key in ("guidance","filter"):
            if config[key]!=support[key]:raise ValueError("Wrong support setting")
        if config.get("conflict_features",False):raise ValueError("Observation experiment must remain separate")
        if not plan["config"]["live_steps"]<=checked["live_transitions"]<plan["config"]["live_steps"]+5120:
            raise ValueError("Experience differs from registered budget")
        provenance=read(training/"provenance.json")
        source={key:provenance[key] for key in ("source_sha256","python","packages")}
        if canonical_source is None:canonical_source=source
        if source!=canonical_source:raise ValueError("Training implementations or environments differ")
        initial=load_evaluation(initial_dir);final=load_evaluation(final_dir)
        if initial[2]!=final[2] or initial[2]!=classical[2]:raise ValueError("Scenarios are not paired")
        for loaded in (initial,final):
            protocol=loaded[1]
            if protocol["seed"]!=plan["evaluation_seed"] or protocol["episodes"]!=plan["evaluation_worlds"]:raise ValueError("Wrong evaluation stream")
            if protocol.get("evaluation_reward_scale",1)!=1 or protocol.get("evaluation_progress_scale",0)!=0:raise ValueError("Non-native evaluation rewards")
            for key in ("algorithm","guidance","filter","action_reference"):
                if protocol[key]!=config[key]:raise ValueError("Evaluation changed "+key)
            if canonical_evaluation is None:canonical_evaluation=protocol["source_sha256"]
            if protocol["source_sha256"]!=canonical_evaluation:raise ValueError("Evaluation implementations differ")
        checkpoint=verified_checkpoint(training,final[1]["checkpoint"])
        if str(checkpoint)!=final[1]["model_path"] or final[1]["checkpoint"]["sha256"]!=checked["model_sha256"]:
            raise ValueError("Evaluated final policy differs from audited checkpoint")
        initial_for(checkpoint,initial[1])
        rows.append({"group":group,"seed":seed,"audit":checked,
                     "initial":{k:float(v.mean()) for k,v in initial[3].items()},
                     "final":{k:float(v.mean()) for k,v in final[3].items()},
                     "initial_csv_sha256":initial[0]["csv_sha256"],"final_csv_sha256":final[0]["csv_sha256"]})
    def statistics(values):
        return {"mean":float(np.mean(values)),"sample_standard_deviation":float(np.std(values,ddof=1)),
                "per_seed":dict(zip(map(str,plan["seeds"]),map(float,values)))}
    grouped={}
    for group in ("g1-f0","g0-f1","g1-f1"):
        selected=sorted([r for r in rows if r["group"]==group],key=lambda r:r["seed"])
        if [r["seed"] for r in selected]!=plan["seeds"]:raise ValueError("Missing or duplicate seed")
        grouped[group]={metric:statistics([r["final"][metric]-r["initial"][metric] for r in selected])
                        for metric in rows[0]["initial"]}
    contrasts={}
    for name,other in (("filter_effect_with_guidance","g1-f0"),("guidance_effect_with_filter","g0-f1")):
        contrasts[name]={}
        for metric in rows[0]["initial"]:
            contrasts[name][metric]=statistics([
                grouped["g1-f1"][metric]["per_seed"][str(seed)]-grouped[other][metric]["per_seed"][str(seed)]
                for seed in plan["seeds"]])
    result={"training_seeds":plan["seeds"],"evaluation_worlds":plan["evaluation_worlds"],"all_registered_rows_included":True,
            "rows":rows,"learning_changes":grouped,"effects_on_learning_changes":contrasts,
            "classical_means":{k:float(v.mean()) for k,v in classical[3].items()},
            "protocol_sha256":hashlib.sha256((PARENT/"protocol.json").read_bytes()).hexdigest(),
            "limitations":plan["limitations"]+["Three support cells only; the four-cell interaction is not estimated.",
                                               "Differences in learning changes are not the same as differences in final system performance.",
                                               "Three seeds provide a limited estimate of training variability; no unseen or official-score claim."]}
    destination.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"results":str(destination),"verified_rows":len(rows),"learning_changes":grouped}))
if __name__=="__main__":main()
