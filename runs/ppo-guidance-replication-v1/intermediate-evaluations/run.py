from pathlib import Path
import hashlib,json,os,subprocess,sys
WORK=Path(__file__).resolve().parent
def main():
    plan=json.loads((WORK/"protocol.json").read_text())
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    for row in plan["rows"]:
        seed=row["seed"];training=Path(row["training"]);folder=training.parent;checkpoint=training/row["checkpoint"]["file"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=row["checkpoint"]["sha256"]:raise ValueError("Checkpoint changed")
        output=folder/"eval-50k-dev20"
        commands=[
            ([sys.executable,"-u","-m","atc_rl.evaluate","--model",str(checkpoint),"--episodes","20","--seed","20260","--out",str(output)],plan["evaluation_source"],f"seed{seed}-evaluation.log"),
            ([sys.executable,"-m","atc_rl.curves","--training-run",str(training),"--classical",plan["classical"],
              "--evaluation",str(folder/"eval-initial-dev20"),"--evaluation",str(output),"--evaluation",str(folder/"eval-final-dev20"),
              "--out",str(folder/"curve-three-points-dev20")],plan["analysis_source"],f"seed{seed}-curve.log")]
        for command,cwd,name in commands:
            print(json.dumps({"stage":name,"checkpoint_live_transitions":row["checkpoint"]["live_transitions"]}),flush=True)
            with (WORK/name).open("x",encoding="utf-8") as stream:
                result=subprocess.run(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,env=env)
            if result.returncode:raise RuntimeError("Stage failed; outputs retained: "+name)
    (WORK/"complete.json").write_text(json.dumps({"intermediate_seeds":[r["seed"] for r in plan["rows"]]}),encoding="utf-8")
if __name__=="__main__":main()
