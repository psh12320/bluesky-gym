"""Verify fresh SAC training and shared native evaluation without scoring claims."""
from pathlib import Path
import csv,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/sac-comparison-source-v1-verify"
sys.path.insert(0,str(SOURCE))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
read=lambda path:json.loads(path.read_text(encoding="utf-8-sig"))

def call(label,module,*args):
    print(json.dumps({"stage":label}),flush=True)
    with (WORK/(label+".log")).open("x",encoding="utf-8") as log:
        result=subprocess.run([sys.executable,"-u","-m",module,*map(str,args)],cwd=SOURCE,
                              env=os.environ,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError("Failed "+label+"; all artifacts retained")

def metrics_match(left,right,agents=10):
    from atc.metrics import METRICS
    def rows(path):
        with path.open(newline="",encoding="utf-8") as stream:
            return {(int(r["episode"]),r["agent"]):r for r in csv.DictReader(stream) if int(r["episode"])<2}
    a,b=rows(left),rows(right)
    assert a.keys()==b.keys() and len(a)==2*agents
    for key,row in a.items():
        assert row["scenario_sha256"]==b[key]["scenario_sha256"]
        for metric in METRICS:assert float(row[metric])==float(b[key][metric]),(key,metric,row,b[key])
    return len(a)*len(METRICS)

def main():
    from atc_rl.cluster import verify
    verify(SOURCE)
    call("sac-train","atc_rl.train_sac","--run-dir",WORK/"sac/train","--workers",1,
         "--live-steps",1280,"--seed",49040,"--collection-steps",64,"--learning-starts",100,
         "--batch-size",64,"--buffer-size",4096,"--checkpoint-live-steps",640,
         "--guidance","--static-filter","--conflict-features","--max-wall-seconds",600)
    call("sac-audit","atc_rl.audit_sac","--run",WORK/"sac/train","--out",WORK/"sac/audit.json")
    for label,filename in (("initial","initial-model.zip"),("trained","model.zip")):
        call(label+"-vector","atc_rl.evaluate","--model",WORK/"sac/train"/filename,
             "--episodes",2,"--seed",20260,"--out",WORK/(label+"-vector"))
    call("trained-native","atc_rl.competition","--env","ma","--model",WORK/"sac/train/model.zip",
         "--episodes",2,"--seed",20260,"--out",WORK/"trained-native")
    old=ROOT/"runs/static-filter-development-v1/ppo"
    matches={"initial_sac_ppo":metrics_match(WORK/"initial-vector/aircraft.csv",old/"eval-initial/aircraft.csv"),
             "trained_sac_vector_native":metrics_match(WORK/"trained-vector/aircraft.csv",WORK/"trained-native/aircraft.csv")}
    call("ppo-regression","atc_rl.evaluate","--model",old/"train/model.zip",
         "--episodes",2,"--seed",20260,"--out",WORK/"ppo-regression")
    matches["unchanged_ppo_evaluation"]=metrics_match(WORK/"ppo-regression/aircraft.csv",old/"eval-final/aircraft.csv")
    call("initial-sa","atc_rl.competition","--env","sa","--model",WORK/"sac/train/initial-model.zip",
         "--episodes",2,"--seed",20260,"--out",WORK/"initial-sa")
    matches["initial_sa_sac_ppo"]=metrics_match(WORK/"initial-sa/aircraft.csv",
         ROOT/"runs/position-scale-development-v1/initial-sa-control/aircraft.csv",1)
    classical=ROOT/"runs/static-filter-development-v1/classical"
    call("comparison","atc_rl.compare","--initial",WORK/"initial-vector","--trained",WORK/"trained-vector",
         "--classical",classical,"--out",WORK/"comparison")
    call("curve","atc_rl.curves","--training-run",WORK/"sac/train","--evaluation",WORK/"initial-vector",
         "--evaluation",WORK/"trained-vector","--classical",classical,"--out",WORK/"curve")
    verify(SOURCE)
    result={"complete":True,"performance_evidence":False,"matched_metric_values":matches,
            "training":read(WORK/"sac/train/training_summary.json"),"audit":read(WORK/"sac/audit.json")}
    with (WORK/"integration-complete.json").open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({"complete":True,"performance_evidence":False,"matched_metric_values":matches}),flush=True)
if __name__=="__main__":main()
