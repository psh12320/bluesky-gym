"""Inspect recorded PPO update budgets without changing models or simulator code."""
from pathlib import Path
import csv,hashlib,io,json,math,zipfile
from datetime import datetime,timezone
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
def main():
    paths=[]
    for item in read(ROOT/"runs/ppo-guidance-replication-v1/three-seed-results.json")["per_seed"]:
        paths.append(Path(item["training_audit"]["run_directory"]))
    for group in ("g0-f1","g1-f1"):
        for seed in (49900,49920,49940):
            paths.append(ROOT/"runs/ppo-filter-ablation-v1"/group/f"seed-{seed}"/"train")
    paths += [ROOT/"runs/ppo-cpa-matched-v1/seed-50400"/arm/"train" for arm in ("features","zeros")]
    paths += [ROOT/"runs"/name/"train" for name in ("goal-offset-scaled-million-retry-v1","mappo-goal-scaled-million-v1")]
    results=[]
    for path in paths:
        if not (path/"training_summary.json").exists():continue
        config,summary=read(path/"config.json"),read(path/"training_summary.json")
        with (path/"learning.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
        changes=[]
        previous_live=previous_updates=previous_epochs=0
        for row in rows:
            live=int(row["live_transitions"]);updates=int(row["optimizer_steps"]);epochs=int(row["policy_epochs"])
            live_delta=live-previous_live
            batches=math.ceil(live_delta/config["batch_size"])
            maximum=batches*config["epochs"]
            actual=updates-previous_updates
            assert 0<=actual<=maximum and 0<epochs-previous_epochs<=config["epochs"]
            changes.append({"live":live_delta,"updates":actual,"maximum_updates":maximum,
                            "epochs_attempted":epochs-previous_epochs,
                            "approx_kl":float(row["approx_kl"]) if row.get("approx_kl") else None,
                            "box_clip":float(row["rollout_action_box_clip_fraction_live"]) if row.get("rollout_action_box_clip_fraction_live") else None})
            previous_live,previous_updates,previous_epochs=live,updates,epochs
        assert previous_live==summary["live_transitions"] and previous_updates==summary["optimizer_steps"]
        def section(values):
            return {"rollouts":len(values),"optimizer_steps":sum(v["updates"] for v in values),
                    "maximum_optimizer_steps_without_KL_stop":sum(v["maximum_updates"] for v in values),
                    "fraction_of_possible_optimizer_steps":sum(v["updates"] for v in values)/sum(v["maximum_updates"] for v in values),
                    "rollouts_stopped_before_full_update_budget":sum(v["updates"]<v["maximum_updates"] for v in values),
                    "rollouts_at_most_two_updates":sum(v["updates"]<=2 for v in values),
                    "mean_logged_approx_kl":float(np.mean([v["approx_kl"] for v in values if v["approx_kl"] is not None]))}
        model=path/"model.zip"
        with zipfile.ZipFile(model) as archive:
            state=torch.load(io.BytesIO(archive.read("policy.pth")),map_location="cpu",weights_only=True)
        std=state["log_std"].exp().tolist()
        results.append({"run":str(path),"status":summary["status"],"seed":config["seed"],"algorithm":config["algorithm"],
                        "live_transitions":summary["live_transitions"],"workers":config["workers"],
                        "guidance":config["guidance"],"filter":config["filter"],"conflict_features":config.get("conflict_features",False),
                        "mask_conflict_features":config.get("mask_conflict_features",False),"initial_action_std":config["initial_action_std"],
                        "final_action_std":std,"latent_heading_std_degrees":[90*config["initial_action_std"],90*std[0]],
                        "all":section(changes),"first10":section(changes[:10]),"last10":section(changes[-10:]),
                        "source_artifacts":{name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in
                         ("config.json","training_summary.json","learning.csv","model.zip")}})
    result={"reviewed_at_utc":datetime.now(timezone.utc).isoformat(),"runs":results,
            "inference":"The installed PPO implementation stops minibatch updates when approximate KL exceeds 1.5 times target_kl. Update-budget shortfall indicates early stopping in these completed rollouts; it is not itself a bug.",
            "limitations":["These runs use different support, feature and worker settings; no pooled causal effect is claimed.",
                           "The logged KL is a mean over the last attempted epoch, not the maximum minibatch KL.",
                           "Latent heading dispersion precedes +/-45-degree command bounds and optional projections.",
                           "Exploration is a candidate explanation, not a demonstrated cause of poor physical performance.",
                           "No model, reward, policy input, scenario or active-run source changed."]}
    with (OUT/"review.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps([{k:r[k] for k in ("run","live_transitions","final_action_std","all","first10","last10")} for r in results],indent=2))
if __name__=="__main__":main()
