"""Run one registered low-rate PPO arm while preserving the running control."""
from datetime import datetime, timezone
from pathlib import Path
import csv, hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def command(module, values):
    args=[sys.executable,"-u","-m",module]
    for key,value in values.items():
        flag="--"+key.replace("_","-")
        if isinstance(value,bool):
            if value:args.append(flag)
        else:args.extend([flag,str(value)])
    return args

def main():
    plan=read(OUT/"protocol.json");source=Path(plan["training_source"]);evaluation=Path(plan["evaluation_source"])
    sys.path.insert(0,str(source))
    from atc_rl.cluster import verify
    from atc_rl.checkpoint_identity import policy_fingerprint,verified_checkpoint
    for directory in (source,evaluation):verify(directory)
    if sha(plan["training_bundle"])!=plan["training_bundle_sha256"] or sha(plan["evaluation_bundle"])!=plan["evaluation_bundle_sha256"]:
        raise ValueError("Frozen source archive changed")
    baseline=Path(plan["baseline_directory"])
    def preserve():
        for name,expected in plan["protected_interrupted_files"].items():
            if sha(ROOT/name)!=expected:raise ValueError("Interrupted experiment artifact changed: "+name)
        for path,expected in ((baseline/"train/config.json",plan["baseline_configuration_sha256"]),
                              (baseline/"train/initial-model.zip",plan["baseline_initial_sha256"]),
                              (plan["baseline_protocol"],plan["baseline_protocol_sha256"]),
                              (plan["pretrained_model"],plan["pretrained_model_sha256"])):
            if sha(path)!=expected:raise ValueError("Preserved control or parent changed")
    preserve();(OUT/"started").mkdir(exist_ok=False)
    protocol_sha,script_sha=sha(OUT/"protocol.json"),sha(Path(__file__))
    environment={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    def execute(args,log,cwd=source):
        preserve()
        if sha(OUT/"protocol.json")!=protocol_sha or sha(Path(__file__))!=script_sha:raise ValueError("Running plan changed")
        print(json.dumps({"stage":str(log.relative_to(OUT)),"at_utc":datetime.now(timezone.utc).isoformat()}),flush=True)
        with log.open("x",encoding="utf-8") as f:result=subprocess.run(args,cwd=cwd,env=environment,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError("Stage failed; preserve "+str(log))
    train=OUT/"train"
    args=command("atc_rl.train",{**plan["config"],"run_dir":train,"pretrained_model":plan["pretrained_model"]})
    (OUT/"launch.json").write_text(json.dumps({"command":args,"protocol_sha256":protocol_sha},indent=2))
    execute(args,OUT/"train.log")
    summary=read(train/"training_summary.json")
    maximum=plan["config"]["workers"]*10*plan["config"]["rollout_steps"]
    if summary["status"]!="complete" or not plan["config"]["live_steps"]<=summary["live_transitions"]<plan["config"]["live_steps"]+maximum:
        raise ValueError("Incomplete registered PPO budget")
    execute(command("atc_rl.audit",{"run":train,"out":OUT/"audit.json"}),OUT/"audit.log")
    if policy_fingerprint(train/"initial-model.zip")!=policy_fingerprint(baseline/"train/initial-model.zip"):
        raise ValueError("Full starting tensors differ from the control")
    if read(OUT/"audit.json")["learning_rate"]!=plan["config"]["learning_rate"]:
        raise ValueError("Serialized optimizer used a different learning rate")
    interrupted=Path(plan["recovery"]["original_directory"])
    with (interrupted/"train/learning.csv").open(newline="",encoding="utf-8") as f:before=list(csv.DictReader(f))
    with (train/"learning.csv").open(newline="",encoding="utf-8") as f:after=list(csv.DictReader(f))
    ignore={"wall_seconds","live_transitions_per_second"}
    for index,row in enumerate(before):
        if {k:v for k,v in row.items() if k not in ignore}!={k:v for k,v in after[index].items() if k not in ignore}:
            raise ValueError("Recovered seeded trajectory differs from the preserved prefix")
    prior_middle=next(r for r in read(interrupted/"train/checkpoints.json") if r["file"].startswith("policy-live-"))
    if policy_fingerprint(train/prior_middle["file"])!=policy_fingerprint(interrupted/"train"/prior_middle["file"]):
        raise ValueError("Recovered 50k checkpoint tensors differ from preserved attempt")
    (OUT/"recovery-prefix-check.json").write_text(json.dumps({
        "matched_rollout_rows_excluding_elapsed_time":len(before),
        "matched_partial_live_transitions":int(before[-1]["live_transitions"]),
        "50k_full_policy_tensors_identical":True,"preserved_partial_model_sha256":plan["recovery"]["original_training_summary"]["model_sha256"],
    },indent=2),encoding="utf-8")
    records=read(train/"checkpoints.json")
    middle=min((r for r in records if r["file"].startswith("policy-live-") and r["live_transitions"]>=50000),key=lambda r:r["live_transitions"])
    if middle["live_transitions"]>=50000+maximum:raise ValueError("Intermediate checkpoint exceeded tolerance")
    selected={"initial":next(r for r in records if r["file"]=="initial-model.zip"),"50k":middle,"final":next(r for r in records if r["file"]=="model.zip")}
    for stage,record in selected.items():
        execute(command("atc_rl.evaluate",{"model":verified_checkpoint(train,record),"episodes":20,"seed":20260,"out":OUT/f"eval-{stage}-dev20"}),OUT/f"eval-{stage}.log",evaluation)
    execute(command("atc_rl.compare",{"initial":OUT/"eval-initial-dev20","trained":OUT/"eval-final-dev20","classical":plan["classical"],"out":OUT/"comparison"}),OUT/"comparison.log")
    args=command("atc_rl.curves",{"training_run":train,"classical":plan["classical"],"out":OUT/"curve"})
    for stage in selected:args.extend(["--evaluation",str(OUT/f"eval-{stage}-dev20")])
    execute(args,OUT/"curve.log")
    preserve()
    for directory in (source,evaluation):verify(directory)
    with (OUT/"complete.json").open("x",encoding="utf-8") as f:json.dump({"finished_at_utc":datetime.now(timezone.utc).isoformat(),"status":"complete","training":summary,"protocol_sha256":protocol_sha,"candidate_promoted":False},f,indent=2)
    print(json.dumps({"status":"complete","paired_control_analysis_required":True}),flush=True)

if __name__=="__main__":
    try:main()
    except BaseException as error:
        with (OUT/"failure.json").open("x",encoding="utf-8") as f:json.dump({"error":repr(error),"all_artifacts_retained":True},f,indent=2)
        raise
