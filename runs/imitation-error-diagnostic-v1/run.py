"""Diagnose preserved imitation errors without fitting or changing a policy."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,sys
import numpy as np
import torch
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/pretrained-ppo-source-v3-verify"
sys.path.insert(0,str(SOURCE))
from atc_rl.demonstrations import load_demonstrations,require_disjoint,sha256
from atc_rl.imitation import actor_mean
from atc_rl.pretrained import inspect_parent
from atc_rl.checkpoint_identity import verified_checkpoint
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))


def stats(prediction,target,mask,world_index):
    if not mask.any():return {"rows":0}
    errors=prediction[mask]-target[mask]
    turns=np.abs(target[mask,0])*45.
    heading=np.abs(errors[:,0])*45.
    speed=np.abs(errors[:,1])
    worlds=world_index[mask]
    return {"rows":int(mask.sum()),"worlds":int(len(np.unique(worlds))),
            "action_mse":float(np.mean(errors.astype(np.float64)**2)),
            "heading_mae_degrees":float(heading.mean()),"speed_action_mae":float(speed.mean()),
            "heading_error_90th_percentile_degrees":float(np.quantile(heading,.9)),
            "teacher_absolute_turn_mean_degrees":float(turns.mean()),
            "predicted_absolute_turn_mean_degrees":float(np.abs(prediction[mask,0]).mean()*45),
            "equal_world_heading_mae_degrees":float(np.mean([heading[worlds==w].mean() for w in np.unique(worlds)])),
            "heading_errors_above_10_degrees":float(np.mean(heading>10)),
            "same_turn_sign":float(np.mean(np.sign(prediction[mask,0])==np.sign(target[mask,0])))}


def main():
    torch.set_num_threads(1)
    parent=ROOT/"runs/imitation-ppo-pilot-v1/bc"
    config,summary,record=inspect_parent(parent/"model.zip")
    model=PPO.load(parent/"model.zip",device="cpu")
    model.policy.set_training_mode(False)
    data={"train":load_demonstrations(ROOT/"runs/imitation-data-v1/train","train"),
          "validation":load_demonstrations(ROOT/"runs/imitation-data-v1/validation","validation")}
    require_disjoint(data["train"],data["validation"])
    result={"recorded_at_utc":datetime.now(timezone.utc).isoformat(),"model_sha256":sha256(parent/"model.zip"),
            "training_updates_performed":0,"datasets":{},"results":{},
            "limitations":["Teacher-visited states only; imitation errors do not establish closed-loop flight performance.",
                "Intervention means the archived teacher intervention flag; it does not isolate static versus traffic decisions.",
                "Correlated decisions are not independent trials; world-balanced descriptive errors are reported without significance claims.",
                "Same-sign accuracy is meaningful for nonzero large turns; zero labels are included in other groups only for completeness.",
                "This analysis does not select a new model or alter either running PPO fine-tuning experiment."]}
    plots=[]
    for name,dataset in data.items():
        a=dataset["arrays"];n=len(a["actor"]);pieces=[]
        with torch.no_grad():
            for start in range(0,n,4096):
                x=torch.as_tensor(a["actor"][start:start+4096],device="cpu")
                pieces.append(actor_mean(model.policy,x).clamp(-1,1).numpy())
        predicted=np.concatenate(pieces);teacher=a["teacher_action"]
        error=(predicted-teacher).astype(np.float64)
        baseline=summary["final_errors"][name]
        np.testing.assert_allclose(np.square(error).mean(),baseline["action_mse"],atol=1e-12,rtol=0)
        turns=np.abs(teacher[:,0])*45
        groups={"all":np.ones(n,dtype=bool),"teacher_intervention":a["intervened"],
                "no_teacher_intervention":~a["intervened"],
                "turn_at_most_2_degrees":turns<=2,
                "turn_2_to_10_degrees":(turns>2)&(turns<=10),
                "turn_above_10_degrees":turns>10,
                "turn_above_20_degrees":turns>20}
        results={}
        for label,mask in groups.items():
            results[label]={"fraction_of_dataset":float(mask.mean()),
                            "learned":stats(predicted,teacher,mask,dataset["world_index"]),
                            "zero_command":stats(np.zeros_like(teacher),teacher,mask,dataset["world_index"]),
                            "nominal_route_command":stats(a["nominal_action"],teacher,mask,dataset["world_index"])}
        component_mse=np.square(error).mean(axis=0)
        zero_component_mse=np.square(teacher.astype(np.float64)).mean(axis=0)
        mse_reductions=zero_component_mse-component_mse
        results["mse_decomposition"]={"learned_heading":float(component_mse[0]),"learned_speed":float(component_mse[1]),
                                      "zero_heading":float(zero_component_mse[0]),"zero_speed":float(zero_component_mse[1]),
                                      "fraction_of_total_mse_reduction_from_speed":float(mse_reductions[1]/mse_reductions.sum())}
        result["results"][name]=results
        result["datasets"][name]={"role":dataset["protocol"]["role"],"seed":dataset["protocol"]["seed"],
            "worlds":len(dataset["scenarios"]),"rows":n,"protocol_sha256":dataset["completion"]["protocol_sha256"],
            "world_manifest_sha256":dataset["completion"]["world_manifest_sha256"]}
        plots.append((name,results))
    assert sha256(parent/"model.zip")==record["sha256"]
    with (OUT/"diagnostic.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=("turn_at_most_2_degrees","turn_2_to_10_degrees","turn_above_10_degrees")
    labels=("Teacher turn <=2°","Teacher turn 2–10°","Teacher turn >10°")
    fig,axes=plt.subplots(1,2,figsize=(11,5.2))
    for ax,(name,values) in zip(axes,plots):
        for offset,(policy,label,color) in zip((-.24,0,.24),(("zero_command","Zero command","#969da7"),
                   ("nominal_route_command","Nominal route command","#b27333"),("learned","Imitation actor","#2b78a3"))):
            ax.bar(np.arange(3)+offset,[values[g][policy]["heading_mae_degrees"] for g in names],width=.23,label=label,color=color)
        ticks=[label+"\n"+str(values[g]["learned"]["rows"])+" decisions" for label,g in zip(labels,names)]
        ax.set_xticks(range(3),ticks);ax.set_ylabel("Heading-command absolute error (degrees)")
        ax.set_title(name.capitalize()+" teacher trajectories");ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc="lower center",ncol=3,bbox_to_anchor=(.5,.11),frameon=False)
    fig.suptitle("Does imitation reproduce the teacher's turns?",fontsize=15)
    fig.text(.5,.047,"Descriptive action errors on teacher-visited states; these are not flight-performance scores.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.07,right=.99,top=.84,bottom=.29,wspace=.26)
    fig.savefig(OUT/"heading-errors.png",dpi=160);fig.savefig(OUT/"heading-errors.pdf");plt.close(fig)
    print(json.dumps({name:{key:result["results"][name][key] for key in ("teacher_intervention","turn_above_10_degrees","mse_decomposition")} for name in result["results"]}))

if __name__=="__main__":main()
