"""Execute the fixed capacity-matched PPO observation ablation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
def main():
    plan=read(PARENT/"protocol.json")
    integration=ROOT/"runs/cpa-mask-development-v1/integration-complete.json"
    curves=ROOT/"runs/ppo-guidance-replication-v1/intermediate-evaluations/complete.json"
    if not integration.is_file() or not curves.is_file():
        raise RuntimeError("Wait for the integration checks and existing intermediate evaluations to finish")
    checked=read(integration)
    if not checked["same_initial_actor_and_critic_tensors"] or not checked["masked_checkpoint_reload_uses_recorded_features"]:
        raise ValueError("Feature control validation did not pass")
    sys.path.insert(0,plan["training_source"])
    from atc_rl.cluster import verify
    from atc_rl.checkpoint_identity import verified_checkpoint,policy_fingerprint
    verify(Path(plan["training_source"]))
    (PARENT/"started").mkdir(exist_ok=False)
    environment={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    def execute(command,cwd,log):
        print(json.dumps({"stage":str(log.relative_to(PARENT)),"at_utc":datetime.now(timezone.utc).isoformat()}),flush=True)
        with log.open("x",encoding="utf-8") as stream:
            result=subprocess.run(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,env=environment)
        if result.returncode:raise RuntimeError(f"Stage failed ({result.returncode}); preserved {log}")
    for seed in plan["seeds"]:
        for arm in plan["arm_order"][str(seed)]:
            folder=PARENT/f"seed-{seed}"/arm;folder.mkdir(parents=True,exist_ok=False)
            values={**plan["config"],**plan["arms"][arm],"seed":seed,"run_dir":str(folder/"train")}
            command=[sys.executable,"-u","-m","atc_rl.train"]
            for key,value in values.items():
                flag="--"+key.replace("_","-")
                if isinstance(value,bool):
                    if value:command.append(flag)
                else:command.extend([flag,str(value)])
            (folder/"launch.json").write_text(json.dumps({"command":command,"protocol_sha256":hashlib.sha256((PARENT/"protocol.json").read_bytes()).hexdigest()},indent=2),encoding="utf-8")
            execute(command,plan["training_source"],folder/"train.log")
            summary=read(folder/"train/training_summary.json")
            if summary["status"]!="complete" or not 100000<=summary["live_transitions"]<102560:
                raise ValueError("Incomplete or invalid training budget; outputs retained")
            execute([sys.executable,"-m","atc_rl.audit","--run",str(folder/"train"),"--out",str(folder/"checkpoint-audit.json")],
                    plan["training_source"],folder/"audit.log")
            records=read(folder/"train/checkpoints.json")
            middle=min((r for r in records if r["file"].startswith("policy-live-") and r["live_transitions"]>=50000),key=lambda r:r["live_transitions"])
            if middle["live_transitions"]>=52560:raise ValueError("Intermediate checkpoint too far from 50k")
            selected={"initial":next(r for r in records if r["file"]=="initial-model.zip"),
                      "50k":middle,"final":next(r for r in records if r["file"]=="model.zip")}
            for stage in plan["stages"]:
                checkpoint=verified_checkpoint(folder/"train",selected[stage])
                execute([sys.executable,"-u","-m","atc_rl.evaluate","--model",str(checkpoint),"--episodes","20","--seed","20260",
                         "--out",str(folder/f"eval-{stage}-dev20")],plan["evaluation_source"],folder/f"eval-{stage}.log")
            execute([sys.executable,"-m","atc_rl.compare","--initial",str(folder/"eval-initial-dev20"),
                     "--trained",str(folder/"eval-final-dev20"),"--classical",plan["classical_evaluation"],"--out",str(folder/"comparison")],
                    plan["evaluation_source"],folder/"comparison.log")
            command=[sys.executable,"-m","atc_rl.curves","--training-run",str(folder/"train"),
                     "--classical",plan["classical_evaluation"],"--out",str(folder/"curve")]
            for stage in plan["stages"]:command+=["--evaluation",str(folder/f"eval-{stage}-dev20")]
            execute(command,plan["evaluation_source"],folder/"curve.log")
            print(json.dumps({"arm_completed":arm,"seed":seed,"live_transitions":summary["live_transitions"]}),flush=True)
        left=PARENT/f"seed-{seed}"/"features/train/initial-model.zip"
        right=PARENT/f"seed-{seed}"/"zeros/train/initial-model.zip"
        if policy_fingerprint(left)!=policy_fingerprint(right):raise ValueError("Paired initial policy tensors differ")
    (PARENT/"complete.json").write_text(json.dumps({"seeds":plan["seeds"],"finished_at_utc":datetime.now(timezone.utc).isoformat()}),encoding="utf-8")
if __name__=="__main__":main()
