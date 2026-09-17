"""Run the registered fresh SAC/PPO pilot; preserve every source and result."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path(__file__).resolve().parent
read=lambda path:json.loads(path.read_text(encoding="utf-8-sig"))


def command(module, values):
    result=[sys.executable,"-u","-m",module]
    for key,value in values.items():
        flag="--"+key.replace("_","-")
        if isinstance(value,bool):
            if value:result.append(flag)
        else:result.extend([flag,str(value)])
    return result


def main():
    plan=read(PARENT/"protocol.json")
    source=Path(plan["source"])
    sys.path.insert(0,str(source))
    from atc_rl.cluster import verify
    from atc_rl.checkpoint_identity import verified_checkpoint
    from atc_rl.compare import load_evaluation
    verify(source)
    digest=hashlib.sha256((ROOT/"runs/sac-comparison-source-v1.zip").read_bytes()).hexdigest()
    if digest!=plan["source_bundle_sha256"]:raise ValueError("Registered source archive changed")
    validation=read(Path(plan["implementation_validation"]))
    if not validation["integration"]["complete"]:raise ValueError("Simulator integration incomplete")
    (PARENT/"started").mkdir(exist_ok=False)
    environment={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    def execute(args,log):
        print(json.dumps({"stage":str(log.relative_to(PARENT)),"at_utc":datetime.now(timezone.utc).isoformat()}),flush=True)
        with log.open("x",encoding="utf-8") as stream:
            result=subprocess.run(args,cwd=source,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f"Stage failed ({result.returncode}); retained {log}")
    for seed in plan["seeds"]:
        for arm in plan["arm_order"][str(seed)]:
            folder=PARENT/f"seed-{seed}"/arm
            folder.mkdir(parents=True,exist_ok=False)
            values={**plan["common"],**plan["arms"][arm],"seed":seed,"run_dir":str(folder/"train")}
            args=command(plan["modules"][arm]["train"],values)
            (folder/"launch.json").write_text(json.dumps({"command":args,"protocol_sha256":hashlib.sha256((PARENT/"protocol.json").read_bytes()).hexdigest()},indent=2),encoding="utf-8")
            execute(args,folder/"train.log")
            summary=read(folder/"train/training_summary.json")
            budget=plan["common"]["live_steps"];maximum=plan["collection_slots"][arm]
            if summary["status"]!="complete" or not budget<=summary["live_transitions"]<budget+maximum:
                raise ValueError("Incomplete or incorrect training budget")
            execute(command(plan["modules"][arm]["audit"],{"run":folder/"train","out":folder/"checkpoint-audit.json"}),folder/"audit.log")
            records=read(folder/"train/checkpoints.json")
            middle=min((r for r in records if r["file"].startswith("policy-live-") and r["live_transitions"]>=50000),key=lambda r:r["live_transitions"])
            if middle["live_transitions"]>=50000+maximum:raise ValueError("Wrong intermediate experience budget")
            selected={"initial":next(r for r in records if r["file"]=="initial-model.zip"),"50k":middle,
                      "final":next(r for r in records if r["file"]=="model.zip")}
            for stage in plan["stages"]:
                model=verified_checkpoint(folder/"train",selected[stage])
                execute(command("atc_rl.evaluate",{"model":model,"episodes":plan["evaluation_worlds"],"seed":plan["evaluation_seed"],"out":folder/f"eval-{stage}-dev20"}),folder/f"eval-{stage}.log")
            execute(command("atc_rl.compare",{"initial":folder/"eval-initial-dev20","trained":folder/"eval-final-dev20","classical":plan["classical_evaluation"],"out":folder/"comparison"}),folder/"comparison.log")
            args=command("atc_rl.curves",{"training_run":folder/"train","classical":plan["classical_evaluation"],"out":folder/"curve"})
            for stage in plan["stages"]:args += ["--evaluation",str(folder/f"eval-{stage}-dev20")]
            execute(args,folder/"curve.log")
            print(json.dumps({"arm_completed":arm,"seed":seed,"live_transitions":summary["live_transitions"]}),flush=True)
        initial={a:load_evaluation(PARENT/f"seed-{seed}"/a/"eval-initial-dev20") for a in ("sac","ppo")}
        import numpy as np
        if initial["sac"][2]!=initial["ppo"][2]:raise ValueError("Initial scenarios differ")
        if any(not np.array_equal(initial["sac"][3][metric],initial["ppo"][3][metric]) for metric in initial["sac"][3]):
            raise ValueError("Initial deterministic physical behavior differs")
        (PARENT/"initialization-check.json").write_text(json.dumps({"identical_initial_physical_metrics":True,"network_tensors_matched":False,"reason":"Algorithm recipes have different critic and actor distribution parameterizations"},indent=2),encoding="utf-8")
    verify(source)
    (PARENT/"complete.json").write_text(json.dumps({"seeds":plan["seeds"],"finished_at_utc":datetime.now(timezone.utc).isoformat()}),encoding="utf-8")
if __name__=="__main__":main()
