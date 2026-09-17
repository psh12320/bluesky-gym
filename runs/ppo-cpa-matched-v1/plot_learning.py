"""Plot every registered conflict-feature seed at its exact saved checkpoint budget."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
OUT=WORK/"three-seed-curves"
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=read(WORK/"paired-three-seed-results.json")
    assert result["all_six_registered_runs_included"] and result["training_seeds"]==[50400,50500,50600]
    OUT.mkdir(exist_ok=False)
    series={};hashes={};rows=[]
    metrics=[("clean_completion","Clean completion change (percentage points)",100),
             ("waypoint_reached","Arrival change (percentage points)",100),
             ("intrusion_time","Conflict-time change (seconds / aircraft)",1),
             ("time_in_restricted_area","Restricted-area-time change (seconds / aircraft)",1)]
    for record in result["rows"]:
        seed=record["seed"]
        for arm in ("features","zeros"):
            path=Path(record["arms"][arm]["curve"])
            curve=read(path);hashes[str(path)]=sha(path)
            assert curve["training_seed"]==seed and len(curve["points"])==3
            points=curve["points"]
            assert points[0]["live_transitions"]==0
            x=np.array([p["live_transitions"] for p in points])
            assert 50000<=x[1]<52560 and 100000<=x[2]<102560
            series[(seed,arm)]={"x":x,"metrics":{}}
            for metric,_,factor in metrics:
                values=np.array([p["metrics"][metric]["mean"] for p in points])
                initial=record["arms"][arm]["initial_means"][metric]
                final=record["arms"][arm]["final_means"][metric]
                assert np.isclose(values[0],initial) and np.isclose(values[-1],final)
                changes=(values-values[0])*factor
                series[(seed,arm)]["metrics"][metric]=changes
                for stage,live,value,change in zip(("initial","50k","final"),x,values,changes):
                    rows.append({"seed":seed,"condition":arm,"checkpoint":stage,"live_transitions":int(live),
                                 "metric":metric,"mean":float(value),"change_from_own_initial":float(change),
                                 "change_unit":"percentage_points" if factor==100 else "seconds_per_aircraft"})
    with (OUT/"points.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.6),layout="constrained")
    colors={"features":"#1766a6","zeros":"#c06a10"}
    labels={"features":"Predictive inputs","zeros":"Masked inputs"}
    for ax,(metric,label,_) in zip(axes.flat,metrics):
        ax.axhline(0,color="#646464",linewidth=1,linestyle="--")
        for arm in ("features","zeros"):
            xs=[];ys=[]
            for seed in result["training_seeds"]:
                case=series[(seed,arm)];x=case["x"]/1000;y=case["metrics"][metric]
                xs.append(x);ys.append(y)
                ax.plot(x,y,color=colors[arm],alpha=.25,linewidth=1)
            ax.errorbar(np.mean(xs,axis=0),np.mean(ys,axis=0),yerr=np.std(ys,axis=0,ddof=1),
                        color=colors[arm],label=labels[arm],linewidth=2,marker="o",markersize=4,capsize=3)
        ax.set_xlabel("Live aircraft transitions (thousands)")
        ax.set_ylabel(label);ax.set_xticks([0,50,100]);ax.grid(alpha=.16)
    axes[0,0].legend(frameon=False)
    fig.suptitle("Conflict-prediction inputs: learning across three training seeds\nThin lines: each seed; thick lines: mean ± sample SD; zero: own untrained policy",fontsize=12)
    fig.supxlabel("20 reused development worlds · Route guidance on, static/traffic filters off · Exact checkpoint counts saved in points.csv",fontsize=9)
    fig.savefig(OUT/"learning-curves.png",dpi=180)
    fig.savefig(OUT/"learning-curves.pdf")
    plt.close(fig)
    with (OUT/"provenance.json").open("x",encoding="utf-8") as f:
        json.dump({"paired_results_sha256":sha(WORK/"paired-three-seed-results.json"),"curves_sha256":hashes,
            "script_sha256":sha(Path(__file__)),"all_six_registered_runs_included":True,
            "error_bars":"Sample standard deviation across three training seeds, not confidence intervals.",
            "x_positions":"Individual curves use exact budgets; group points use mean exact budget at the registered stage.",
            "unseen_scenarios_used":False},f,indent=2)
    print(json.dumps({"figure":str(OUT/"learning-curves.png"),"data_points":len(rows)}))
if __name__=="__main__":main()
