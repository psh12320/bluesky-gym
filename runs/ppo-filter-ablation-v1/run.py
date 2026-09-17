"""Run one precommitted filter-ablation cohort without changing frozen code."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group",choices=["g0-f1","g1-f1"])
    args=parser.parse_args()
    protocol=read(PARENT/"protocol.json")
    group=next(g for g in protocol["support_groups"] if g["name"]==args.group)
    cohort=PARENT/args.group
    cohort.mkdir(exist_ok=False)
    environment={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    def execute(command,cwd,log):
        print(json.dumps({"stage":str(log.relative_to(PARENT)),"started_at_utc":datetime.now(timezone.utc).isoformat()}),flush=True)
        with log.open("x",encoding="utf-8") as stream:
            result=subprocess.run(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,env=environment)
        if result.returncode:raise RuntimeError(f"Stage failed ({result.returncode}); preserved log: {log}")
    for seed in protocol["seeds"]:
        folder=cohort/f"seed-{seed}"
        folder.mkdir()
        command=[sys.executable,"-u","-m","atc_rl.train"]
        values={**protocol["config"],"seed":seed,"guidance":group["guidance"],"filter":group["filter"],"run_dir":str(folder/"train")}
        for key,value in values.items():
            flag="--"+key.replace("_","-")
            if isinstance(value,bool):
                if value:command.append(flag)
            else:command.extend([flag,str(value)])
        (folder/"launch.json").write_text(json.dumps({"command":command,"protocol_sha256":hashlib.sha256((PARENT/"protocol.json").read_bytes()).hexdigest()},indent=2),encoding="utf-8")
        execute(command,protocol["training_source"],folder/"train.log")
        summary=read(folder/"train/training_summary.json")
        if summary["status"]!="complete" or not protocol["config"]["live_steps"]<=summary["live_transitions"]<protocol["config"]["live_steps"]+5120:
            raise RuntimeError("Incomplete or invalid training budget; outputs preserved")
        execute([sys.executable,"-m","atc_rl.audit","--run",str(folder/"train"),"--out",str(folder/"checkpoint-audit.json")],
                protocol["training_source"],folder/"audit.log")
        for stage,filename in (("initial","initial-model.zip"),("final","model.zip")):
            execute([sys.executable,"-u","-m","atc_rl.evaluate","--model",str(folder/"train"/filename),
                     "--episodes",str(protocol["evaluation_worlds"]),"--seed",str(protocol["evaluation_seed"]),
                     "--out",str(folder/f"eval-{stage}-dev20")],protocol["evaluation_source"],folder/f"eval-{stage}.log")
        execute([sys.executable,"-m","atc_rl.compare","--initial",str(folder/"eval-initial-dev20"),"--trained",str(folder/"eval-final-dev20"),
                 "--classical",protocol["classical_evaluation"],"--out",str(folder/"comparison")],
                protocol["evaluation_source"],folder/"comparison.log")
        execute([sys.executable,"-m","atc_rl.curves","--training-run",str(folder/"train"),"--classical",protocol["classical_evaluation"],
                 "--evaluation",str(folder/"eval-initial-dev20"),"--evaluation",str(folder/"eval-final-dev20"),
                 "--out",str(folder/"curve")],protocol["training_source"],folder/"curve.log")
        comparison=read(folder/"comparison/comparison.json")
        print(json.dumps({"seed_completed":seed,"group":args.group,"live_transitions":summary["live_transitions"],
                          "initial":comparison["means"]["initial"],"final":comparison["means"]["trained"]}),flush=True)
    (cohort/"complete.json").write_text(json.dumps({"seeds":protocol["seeds"],"finished_at_utc":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
if __name__=="__main__":main()
