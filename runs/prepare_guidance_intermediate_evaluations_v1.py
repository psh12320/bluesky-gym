from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
root=Path.cwd();parent=root/'runs/ppo-guidance-replication-v1';work=parent/'intermediate-evaluations'
work.mkdir(exist_ok=False)
rows=[]
for seed in (49920,49940):
    training=parent/f'seed-{seed}'/'train'
    manifest=json.loads((training/'checkpoints.json').read_text())
    chosen=min((row for row in manifest if row['file'].startswith('policy-live-') and row['live_transitions']>=50000),key=lambda row:row['live_transitions'])
    assert 50000<=chosen['live_transitions']<55120
    assert hashlib.sha256((training/chosen['file']).read_bytes()).hexdigest()==chosen['sha256']
    rows.append({'seed':seed,'training':str(training),'checkpoint':chosen})
protocol={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'selection':'First saved checkpoint at/above 50k; selected before these intermediate evaluations.',
          'rows':rows,'evaluation_worlds':20,'evaluation_seed':20260,
          'evaluation_source':str(root/'runs/onpolicy-goal-offset-source-v1-verify'),
          'analysis_source':str(root/'runs/onpolicy-learning-source-v2-verify'),
          'classical':str(root/'runs/ppo-direct-pilot-v1/eval-classical-dev20'),
          'first_seed_curve':'Existing seed49900 curve includes initial, 50100 and final; preserve it.'}
(work/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
driver=r'''from pathlib import Path
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
'''
compile(driver,'run.py','exec')
(work/'run.py').write_text(driver,encoding='utf-8')
print(json.dumps({'preselected':[{key:row[key] for key in ('seed','checkpoint')} for row in rows],'evaluations_started':False}))
