from pathlib import Path
import csv,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/onpolicy-cpa-source-v2-verify"
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
sys.path.insert(0,str(SOURCE))
def call(label,module,*args):
    print(json.dumps({"stage":label}),flush=True)
    with (WORK/(label+".log")).open("x",encoding="utf-8") as stream:
        result=subprocess.run([sys.executable,"-u","-m",module,*map(str,args)],cwd=SOURCE,stdout=stream,stderr=subprocess.STDOUT,env=os.environ)
    if result.returncode:raise RuntimeError("Stage failed; preserved "+label+".log")
def main():
    from atc_rl.cluster import verify
    from atc_rl.checkpoint_identity import policy_fingerprint
    verify(SOURCE)
    for arm in ("features","zeros"):
        flags=["--mask-conflict-features"] if arm=="zeros" else []
        call(arm+"-train","atc_rl.train","--algorithm","ppo","--workers","1","--live-steps","1280",
             "--rollout-steps","64","--batch-size","256","--epochs","2","--seed","49020","--device","cpu",
             "--guidance","--conflict-features",*flags,"--action-reference","goal_offset","--neutral-action-mean",
             "--initial-action-std",".05","--reward-scale",".01","--checkpoint-live-steps","1280",
             "--max-wall-seconds","600","--run-dir",WORK/arm/"train")
        call(arm+"-audit","atc_rl.audit","--run",WORK/arm/"train","--out",WORK/arm/"audit.json")
    fingerprint=policy_fingerprint(WORK/"features/train/initial-model.zip")
    if fingerprint!=policy_fingerprint(WORK/"zeros/train/initial-model.zip"):raise ValueError("Initial tensors differ")
    audits=[json.loads((WORK/arm/"audit.json").read_text()) for arm in ("features","zeros")]
    if any(a["status"]!="complete" for a in audits):raise ValueError("Incomplete smoke training")
    for key in ("actor_parameter_count","critic_parameter_count"):
        if audits[0][key]!=audits[1][key]:raise ValueError("Network capacity differs")
    for stage,filename in (("initial","initial-model.zip"),("trained","model.zip")):
        call("zeros-"+stage+"-evaluation","atc_rl.evaluate","--model",WORK/"zeros/train"/filename,
             "--episodes","2","--seed","20260","--out",WORK/"zeros"/("eval-"+stage))
        protocol=json.loads((WORK/"zeros"/("eval-"+stage)/"protocol.json").read_text())
        assert protocol["conflict_features"] is True and protocol["mask_conflict_features"] is True
    original=json.loads((ROOT/"runs/cpa-observation-development-v1/g1-cpa1.json").read_text())
    reference={(row["episode"],f'KL00{row["agent"]+1}'):row for row in original["rows"]}
    with (WORK/"zeros/eval-initial/aircraft.csv").open(newline="") as stream:observed=list(csv.DictReader(stream))
    metrics=("waypoint_reached","flight_time","intrusion_events","intrusion_time","restricted_area_events",
             "time_in_restricted_area","sector_exit_events","time_outside_sector","total_reward")
    assert len(observed)==len(reference)==20
    for row in observed:
        expected=reference[int(row["episode"]),row["agent"]]
        for key in metrics:assert float(row[key])==expected[key],key
    scenarios=json.loads((WORK/"zeros/eval-initial/scenarios.json").read_text())
    assert [row["sha256"] for row in scenarios]==original["scenarios"]
    verify(SOURCE)
    record={"same_initial_actor_and_critic_tensors":True,"initial_policy_fingerprint":fingerprint,
            "actor_parameters_each":audits[0]["actor_parameter_count"],"critic_parameters_each":audits[0]["critic_parameter_count"],
            "smoke_live_transitions_by_arm":{a["run_directory"]:a["live_transitions"] for a in audits},
            "masked_initial_physical_metrics_identical":180,"worlds":2,
            "masked_checkpoint_reload_uses_recorded_features":True,
            "performance_evidence":False,"scope":"Small integration runs, excluded from pilot performance claims."}
    (WORK/"integration-complete.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record),flush=True)
if __name__=="__main__":main()
