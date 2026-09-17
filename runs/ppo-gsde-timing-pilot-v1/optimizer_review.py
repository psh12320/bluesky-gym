"""Account for adaptive optimization in the two completed gSDE training arms."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,math
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))

def main():
    destination=PARENT/"optimizer-review.json"
    if destination.exists():raise FileExistsError(destination)
    plan=read(PARENT/"protocol.json")
    results={}
    for arm in ("every1","every12"):
        folder=PARENT/"seed-51700"/arm/"train"
        summary=read(folder/"training_summary.json");config=read(folder/"config.json")
        if summary["status"]!="complete":raise ValueError("Training not complete")
        if hashlib.sha256((folder/"model.zip").read_bytes()).hexdigest()!=summary["model_sha256"]:
            raise ValueError("Final model identity changed")
        expected={**plan["config"],**plan["arms"][arm],"seed":51700}
        if any(config.get(k)!=v for k,v in expected.items()):raise ValueError("Unregistered configuration")
        path=folder/"learning.csv"
        with path.open(newline="",encoding="utf-8") as stream:rows=list(csv.DictReader(stream))
        prior_live=prior_updates=0;details=[]
        for row in rows:
            live=int(row["live_transitions"]);updates=int(row["optimizer_steps"])
            collected=live-prior_live
            if collected!=int(row["rollout_live_samples"]):raise ValueError("Rollout live sample count differs")
            planned=config["epochs"]*math.ceil(collected/config["batch_size"])
            applied=updates-prior_updates
            if not 0<=applied<=planned:raise ValueError("Optimizer count outside possible epoch budget")
            details.append({"rollout":int(row["rollout"]),"live_samples":collected,
                            "planned_steps_without_early_stop":planned,"applied_steps":applied,
                            "below_full_budget":applied<planned,"logged_mean_approximate_kl":float(row["approx_kl"])})
            prior_live,prior_updates=live,updates
        if prior_updates!=summary["optimizer_steps"] or prior_live!=summary["live_transitions"]:
            raise ValueError("Learning counters differ from training summary")
        planned=sum(r["planned_steps_without_early_stop"] for r in details)
        results[arm]={"live_transitions":prior_live,"applied_optimizer_steps":prior_updates,
                     "planned_steps_without_early_stop":planned,"applied_fraction":prior_updates/planned,
                     "rollouts_below_full_budget":sum(r["below_full_budget"] for r in details),
                     "rollouts":len(details),"target_kl":config["target_kl"],"epochs":config["epochs"],
                     "learning_csv_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"rollout_details":details}
    import stable_baselines3.ppo.ppo as ppo_module
    implementation=Path(ppo_module.__file__)
    result={"recorded_at_utc":datetime.now(timezone.utc).isoformat(),"arms":results,
        "sb3_ppo_implementation_sha256":hashlib.sha256(implementation.read_bytes()).hexdigest(),
        "mechanism":"Frozen PPO stops an epoch before optimization when the current minibatch approximate KL exceeds 1.5 times target_kl. Logged KL is an average, so it is not used to identify the triggering minibatch.",
        "interpretation":"The registered timing intervention also changes the amount of optimization through adaptive KL stopping. Report these counts with performance; the study does not isolate physical exploration persistence from this optimizer response.",
        "changes_to_running_study":False,"policy_quality_assessed":False}
    with destination.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({arm:{k:v for k,v in result.items() if k!="rollout_details"} for arm,result in results.items()}))
if __name__=="__main__":main()
