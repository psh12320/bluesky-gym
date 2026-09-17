"""Native harness parity checks; excluded from performance/model selection."""
from pathlib import Path
import csv,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/onpolicy-native-source-v2-verify"
OLD=ROOT/"runs/onpolicy-static-source-v1-verify"
sys.path.insert(0,str(SOURCE))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
def call(label,module,*args,source=SOURCE):
    print(json.dumps({"stage":label}),flush=True)
    with (WORK/(label+".log")).open("x",encoding="utf-8") as f:
        result=subprocess.run([sys.executable,"-u","-m",module,*map(str,args)],cwd=source,
                              stdout=f,stderr=subprocess.STDOUT,env=os.environ)
    if result.returncode:raise RuntimeError("Stage failed; retained "+label+".log")
def compare(observed,reference):
    from atc.metrics import METRICS
    def rows(path):
        with path.open(newline="",encoding="utf-8") as f:
            return {(int(r["episode"]),r["agent"]):r for r in csv.DictReader(f) if int(r["episode"])<2}
    left,right=rows(observed),rows(reference)
    assert len(left)==len(right)==20 and left.keys()==right.keys()
    for identity,row in left.items():
        expected=right[identity]
        assert row["scenario_sha256"]==expected["scenario_sha256"]
        for key in METRICS:
            if float(row[key])!=float(expected[key]):
                raise ValueError(f"Native metric changed: {identity} {key}: {row[key]} != {expected[key]}")
    return len(left)*len(METRICS)
def main():
    from atc_rl.cluster import verify
    verify(SOURCE);verify(OLD)
    static=ROOT/"runs/static-filter-development-v1"
    masked=ROOT/"runs/cpa-mask-development-v1/zeros"
    unsupported=ROOT/"runs/goal-offset-scaled-million-retry-v1"
    filtered=ROOT/"runs/ppo-filter-ablation-v1/g1-f1/seed-49940"
    mappo=static/"mappo/train/model.zip"
    cases=[
        ("ppo-static",static/"ppo/train/model.zip",static/"ppo/eval-final/aircraft.csv"),
        ("mappo-static",mappo,ROOT/"runs/native-policy-development-v1/mappo-reference/aircraft.csv"),
        ("ppo-masked",masked/"train/model.zip",masked/"eval-trained/aircraft.csv"),
        ("ppo-unsupported",unsupported/"train/model.zip",unsupported/"eval-final-dev20/aircraft.csv"),
        ("ppo-filtered",filtered/"train/model.zip",filtered/"eval-final-dev20/aircraft.csv")]
    transfers=[]
    for name,model in [("ppo-static",static/"ppo/train/model.zip"),("ppo-unsupported",unsupported/"train/model.zip")]:
        destination=WORK/("sa-"+name)
        call("sa-"+name,"atc_rl.competition","--env","sa","--model",model,"--episodes",2,
             "--seed",20260,"--out",destination)
        summary=read(destination/"summary.json")
        assert summary["agent_episodes"]==2 and summary["episodes"]==2
        assert summary["runtime"]["simulated_scripted_intruders"]==10
        assert summary["runtime"]["observed_traffic_slots"]==9
        assert summary["runtime"]["single_agent_transfer"]
        assert all(s["simulated_aircraft_at_reset"]==11 for s in read(destination/"scenarios.json"))
        transfers.append({"case":name,"runtime":summary["runtime"],"csv_sha256":summary["csv_sha256"]})
    results=[]
    for name,model,reference in cases:
        destination=WORK/("ma-"+name)
        call("ma-"+name,"atc_rl.competition","--env","ma","--model",model,"--episodes",2,
             "--seed",20260,"--out",destination)
        matched=compare(destination/"aircraft.csv",reference)
        summary=read(destination/"summary.json")
        assert summary["agent_episodes"]==20 and not summary["official_protocol"]
        assert not summary["actor_inference_uses_joint_context"]
        assert all(s["simulated_aircraft_at_reset"]==10 for s in read(destination/"scenarios.json"))
        results.append({"case":name,"matched_physical_metrics":matched,
                        "model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
                        "runtime":summary["runtime"],"csv_sha256":summary["csv_sha256"]})
        (WORK/(name+"-parity.json")).write_text(json.dumps(results[-1],indent=2),encoding="utf-8")
    verify(SOURCE);verify(OLD)
    record={"ma_cases":results,"matched_ma_metrics_total":sum(r["matched_physical_metrics"] for r in results),
            "single_agent_transfers":transfers,"original_scoring_records_identical_to_trace":True,
            "original_harness_sha256":hashlib.sha256((SOURCE/"scripts/evaluate_competition.py").read_bytes()).hexdigest(),
            "performance_evidence":False,"official_or_unseen_scenarios_used":False}
    with (WORK/"integration-complete.json").open("x",encoding="utf-8") as f:json.dump(record,f,indent=2)
    print(json.dumps({"matched_ma_metrics_total":record["matched_ma_metrics_total"],
                     "single_agent_transfer_cases":len(transfers),"performance_evidence":False}),flush=True)
if __name__=="__main__":main()
