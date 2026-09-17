
"""Simulator verification of observation-only traffic-position rescaling."""
from pathlib import Path
import csv,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/onpolicy-position-source-v1-verify"
sys.path.insert(0,str(SOURCE))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))

def call(label,module,*args):
    print(json.dumps({"stage":label}),flush=True)
    with (WORK/(label+".log")).open("x",encoding="utf-8") as log:
        result=subprocess.run([sys.executable,"-u","-m",module,*map(str,args)],cwd=SOURCE,
                              env=os.environ,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError("Failed "+label+"; outputs retained")

def metrics_match(left,right,agents):
    from atc.metrics import METRICS
    def read_rows(p):
        with p.open(newline="",encoding="utf-8") as f:
            return {(int(r["episode"]),r["agent"]):r for r in csv.DictReader(f) if int(r["episode"])<2}
    a,b=read_rows(left),read_rows(right)
    assert a.keys()==b.keys() and len(a)==2*agents
    for key,row in a.items():
        assert row["scenario_sha256"]==b[key]["scenario_sha256"]
        for metric in METRICS:assert float(row[metric])==float(b[key][metric]),(key,metric,row,b[key])
    return len(a)*len(METRICS)

def main():
    from atc_rl.cluster import verify
    from atc_rl.checkpoint_identity import policy_fingerprint
    from atc_rl.algorithm_compare import actor_match
    verify(SOURCE)
    old=ROOT/"runs/static-filter-development-v1/ppo/train"
    config=read(old/"config.json")
    fields=("workers","live_steps","rollout_steps","batch_size","epochs","seed","initial_action_std",
            "neutral_action_mean","action_reference","guidance","filter","static_filter","conflict_features",
            "mask_conflict_features","reward_scale","progress_scale","checkpoint_live_steps","max_wall_seconds")
    base=[]
    for key in fields:
        value=config[key]
        if isinstance(value,bool):
            if value:base.append("--"+key.replace("_","-"))
        else:base+=["--"+key.replace("_","-"),str(value)]
    call("control-train","atc_rl.train","--algorithm","ppo",*base,"--run-dir",WORK/"control/train")
    for filename in ("initial-model.zip","model.zip"):
        assert policy_fingerprint(old/filename)==policy_fingerprint(WORK/"control/train"/filename)
    for algorithm in ("ppo","mappo"):
        args=["--algorithm",algorithm,*base,"--traffic-position-scale",20,"--run-dir",WORK/algorithm/"train"]
        if algorithm=="mappo":args+=["--actor-reference",WORK/"ppo/train/initial-model.zip"]
        call(algorithm+"-train","atc_rl.train",*args)
        call(algorithm+"-audit","atc_rl.audit","--run",WORK/algorithm/"train","--out",WORK/algorithm/"audit.json")
    assert policy_fingerprint(WORK/"control/train/initial-model.zip")==policy_fingerprint(WORK/"ppo/train/initial-model.zip")
    identity=actor_match(WORK/"ppo/train/initial-model.zip",WORK/"mappo/train/initial-model.zip")
    matches={}
    initial=WORK/"ppo/train/initial-model.zip"
    trained=WORK/"ppo/train/model.zip"
    call("initial-ma","atc_rl.competition","--env","ma","--model",initial,"--episodes",2,"--seed",20260,"--out",WORK/"initial-ma")
    reference=ROOT/"runs/ppo-exploration-static-pilot-v1/seed-51200/std005/eval-initial-dev20/aircraft.csv"
    matches["unchanged_initial_ma"]=metrics_match(WORK/"initial-ma/aircraft.csv",reference,10)
    call("trained-vector","atc_rl.evaluate","--model",trained,"--episodes",2,"--seed",20260,"--out",WORK/"trained-vector")
    call("trained-native","atc_rl.competition","--env","ma","--model",trained,"--episodes",2,"--seed",20260,"--out",WORK/"trained-native")
    matches["trained_vector_native"]=metrics_match(WORK/"trained-vector/aircraft.csv",WORK/"trained-native/aircraft.csv",10)
    for label,model in (("initial-sa-control",old/"initial-model.zip"),("initial-sa-scaled",initial)):
        call(label,"atc_rl.competition","--env","sa","--model",model,"--episodes",2,"--seed",20260,"--out",WORK/label)
    matches["unchanged_initial_sa"]=metrics_match(WORK/"initial-sa-control/aircraft.csv",WORK/"initial-sa-scaled/aircraft.csv",1)
    for label in ("initial-ma","trained-vector","trained-native","initial-sa-scaled"):
        summary=read(WORK/label/"summary.json")
        assert summary["runtime"]["traffic_position_scale"]==20
        assert read(WORK/label/"protocol.json")["traffic_position_scale"]==20
    verify(SOURCE)
    result={"gaussian_default_initial_and_trained_policy_tensors_unchanged":True,
        "scaled_and_unscaled_initial_full_policy_tensors_identical":True,
        "ppo_mappo_initial_actor_match":identity,"matched_metric_values":matches,"performance_evidence":False,
        "training":{a:read(WORK/a/"train/training_summary.json") for a in ("ppo","mappo")}}
    with (WORK/"integration-complete.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"complete":True,"matched_metrics":matches,"performance_evidence":False}),flush=True)

if __name__=="__main__":main()
