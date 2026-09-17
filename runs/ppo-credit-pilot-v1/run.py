"""Run only the new GAE arm; reuse the separately registered PPO control."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))

def main():
    plan=read(PARENT/"protocol.json");source=Path(plan["source"])
    sys.path.insert(0,str(source))
    from atc_rl.cluster import verify
    from atc_rl.checkpoint_identity import verified_checkpoint
    verify(source)
    if hashlib.sha256((ROOT/"runs/sac-comparison-source-v1.zip").read_bytes()).hexdigest()!=plan["source_bundle_sha256"]:
        raise ValueError("Frozen source changed")
    if hashlib.sha256(Path(plan["control_protocol"]).read_bytes()).hexdigest()!=plan["control_protocol_sha256"]:
        raise ValueError("Registered control changed")
    (PARENT/"started").mkdir(exist_ok=False)
    folder=Path(plan["new_directory"]);folder.mkdir(parents=True,exist_ok=False)
    values={**plan["config"],**plan["arms"]["lambda099"],"seed":plan["seed"],"run_dir":str(folder/"train")}
    command=[sys.executable,"-u","-m","atc_rl.train"]
    for key,value in values.items():
        flag="--"+key.replace("_","-")
        if isinstance(value,bool):
            if value:command.append(flag)
        else:command += [flag,str(value)]
    (folder/"launch.json").write_text(json.dumps({"command":command,"protocol_sha256":hashlib.sha256((PARENT/"protocol.json").read_bytes()).hexdigest()},indent=2),encoding="utf-8")
    environment={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    def execute(args,log):
        print(json.dumps({"stage":str(log.relative_to(PARENT)),"at_utc":datetime.now(timezone.utc).isoformat()}),flush=True)
        with log.open("x",encoding="utf-8") as stream:
            result=subprocess.run(args,cwd=source,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f"Stage failed ({result.returncode}); retained {log}")
    execute(command,folder/"train.log")
    summary=read(folder/"train/training_summary.json")
    if summary["status"]!="complete" or not 100000<=summary["live_transitions"]<102560:
        raise ValueError("Incomplete or wrong registered training budget")
    execute([sys.executable,"-m","atc_rl.audit","--run",str(folder/"train"),"--out",str(folder/"checkpoint-audit.json")],folder/"audit.log")
    records=read(folder/"train/checkpoints.json")
    middle=min((r for r in records if r["file"].startswith("policy-live-") and r["live_transitions"]>=50000),key=lambda r:r["live_transitions"])
    if middle["live_transitions"]>=52560:raise ValueError("Wrong intermediate experience budget")
    selected={"initial":next(r for r in records if r["file"]=="initial-model.zip"),"50k":middle,"final":next(r for r in records if r["file"]=="model.zip")}
    for stage in plan["stages"]:
        model=verified_checkpoint(folder/"train",selected[stage])
        execute([sys.executable,"-u","-m","atc_rl.evaluate","--model",str(model),"--episodes","20","--seed","20260","--out",str(folder/f"eval-{stage}-dev20")],folder/f"eval-{stage}.log")
    execute([sys.executable,"-m","atc_rl.compare","--initial",str(folder/"eval-initial-dev20"),"--trained",str(folder/"eval-final-dev20"),"--classical",plan["classical_evaluation"],"--out",str(folder/"comparison")],folder/"comparison.log")
    command=[sys.executable,"-m","atc_rl.curves","--training-run",str(folder/"train"),"--classical",plan["classical_evaluation"],"--out",str(folder/"curve")]
    for stage in plan["stages"]:command += ["--evaluation",str(folder/f"eval-{stage}-dev20")]
    execute(command,folder/"curve.log")
    verify(source)
    (PARENT/"lambda099-complete.json").write_text(json.dumps({"seed":plan["seed"],"finished_at_utc":datetime.now(timezone.utc).isoformat(),"paired_analysis_requires_control_completion":True}),encoding="utf-8")
if __name__=="__main__":main()
