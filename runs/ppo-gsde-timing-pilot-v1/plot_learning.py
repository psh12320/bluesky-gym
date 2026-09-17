"""Plot both completed exploration arms and retain exact checkpoint provenance."""
from pathlib import Path
import csv,hashlib,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
PARENT=Path(__file__).resolve().parent

def main():
    source=PARENT/"paired-results.json"
    results=json.loads(source.read_text())
    destination=PARENT/"paired-curves"
    destination.mkdir(exist_ok=False)
    panels=[("clean_completion","Clean completion (%)",100),
            ("flight_time","Flight time (seconds)",1),
            ("intrusion_time","Intrusion time (seconds)",1),
            ("time_in_restricted_area","Restricted-area time (seconds)",1)]
    styles={"every1":("Every decision","#3268a8"),"every12":("Every 12 decisions","#bd6532")}
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.5),layout="constrained")
    table=[]
    for ax,(metric,title,scale) in zip(axes.flat,panels):
        for arm,(label,color) in styles.items():
            points=results["arms"][arm]["curve"]["points"]
            x=np.array([p["live_transitions"] for p in points])/1000
            y=np.array([p["metrics"][metric]["mean"] for p in points])*scale
            bands=np.array([p["metrics"][metric]["world_bootstrap_95_interval"] for p in points])*scale
            ax.plot(x,y,"o-",label=label,color=color,linewidth=2)
            ax.fill_between(x,bands[:,0],bands[:,1],alpha=.12,color=color)
            for point in points:
                table.append({"arm":arm,"live_transitions":point["live_transitions"],"metric":metric,
                              "mean":point["metrics"][metric]["mean"],
                              "world_bootstrap_lower":point["metrics"][metric]["world_bootstrap_95_interval"][0],
                              "world_bootstrap_upper":point["metrics"][metric]["world_bootstrap_95_interval"][1],
                              "policy_fingerprint":point["policy_fingerprint"]})
        benchmark=results["arms"]["every1"]["curve"]["classical_means"][metric]*scale
        ax.axhline(benchmark,linestyle="--",color="#557548",label="Fixed classical system",linewidth=1.5)
        ax.set_title(title);ax.set_xlabel("Live training transitions (thousands)")
        ax.set_xticks([0,50,100]);ax.set_xlim(-3,104);ax.grid(alpha=.2)
        if scale==100:ax.set_ylim(0,100)
        else:ax.set_ylim(bottom=0)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("PPO exploration persistence: one training seed, 20 development worlds\n"
                 "Shading: world-bootstrap 95% intervals; classical system includes traffic filtering",fontsize=12)
    fig.savefig(destination/"learning-curves.png",dpi=170)
    fig.savefig(destination/"learning-curves.pdf")
    plt.close(fig)
    with (destination/"points.csv").open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    record={"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "point_count":len(table),"seed":results["seed"],"fresh_seed_replications_included":False,
            "noise_schedule_note":"Both schedules also resample at rollout starts.",
            "optimizer_review":"../optimizer-review.json","decision":"../decision.json",
            "limitations":results["limitations"]}
    (destination/"provenance.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps({"figure":str(destination/"learning-curves.png"),"points":len(table)}))
if __name__=="__main__":main()
