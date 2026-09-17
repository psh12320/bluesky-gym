"""Validate observation-only simulator parity and a tiny saved-policy training cycle."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/onpolicy-cpa-source-v1-verify"
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
sys.path.insert(0,str(SOURCE))
def execute(module,*args):
    command=[sys.executable,"-u","-m",module,*map(str,args)]
    label=module.rsplit(".",1)[-1]
    with (WORK/(label+"-integration.log")).open("x",encoding="utf-8") as stream:
        result=subprocess.run(command,cwd=SOURCE,stdout=stream,stderr=subprocess.STDOUT,env=os.environ)
    if result.returncode:raise RuntimeError(f"Failed integration stage {label}; log preserved")
def main():
    import numpy as np
    from atc_rl.cluster import verify
    from atc_rl.world_pool import WorldPool
    verify(SOURCE)
    cases={}
    for guidance in (False,True):
        for enabled in (False,True):
            label=f"g{int(guidance)}-cpa{int(enabled)}"
            print(json.dumps({"parity_case_started":label}),flush=True)
            env=WorldPool(1,WORK/label,guidance=guidance,filter=False,action_reference="goal_offset",conflict_features=enabled)
            rows=[];fingerprints=[];decisions=0;episodes=0
            try:
                env.seed(20260);observation=env.reset()
                fingerprint=env.reset_infos[0]["scenario_sha256"]
                actor_dim=env.observation_space["actor"].shape[0]
                while episodes<2:
                    observation,rewards,dones,infos=env.step(np.zeros((10,2),dtype=np.float32))
                    decisions+=1
                    assert np.isfinite(observation["actor"]).all()
                    for index,info in enumerate(infos):
                        if info["aircraft_done"]:
                            rows.append({"episode":episodes,"agent":index,**info["metrics"]})
                    if infos[0]["world_completed"]:
                        fingerprints.append(fingerprint);episodes+=1
                        fingerprint=env.reset_infos[0]["scenario_sha256"]
                assert len(rows)==20
                cases[label]={"actor_dim":actor_dim,"decisions":decisions,"scenarios":fingerprints,"rows":rows}
            finally:env.close()
            (WORK/(label+".json")).write_text(json.dumps(cases[label],indent=2),encoding="utf-8")
    for guidance in (False,True):
        left=cases[f"g{int(guidance)}-cpa0"];right=cases[f"g{int(guidance)}-cpa1"]
        assert left["rows"]==right["rows"] and left["scenarios"]==right["scenarios"] and left["decisions"]==right["decisions"]
        assert right["actor_dim"]-left["actor_dim"]==45
    record={"cases":4,"worlds_per_case":2,"all_scored_aircraft_metrics_identical":True,
            "scenarios_identical":True,"added_local_features":45,"actor_dimensions":{k:v["actor_dim"] for k,v in cases.items()},
            "scope":"Fixed zero latent actions, goal_offset, guidance off/on, filtering off; validates observation-only parity, not policy quality."}
    (WORK/"simulator-parity.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record),flush=True)
    execute("atc_rl.train","--algorithm","ppo","--workers","1","--live-steps","1280","--rollout-steps","64",
            "--batch-size","256","--epochs","2","--seed","49010","--device","cpu","--conflict-features",
            "--action-reference","goal_offset","--neutral-action-mean","--initial-action-std",".05",
            "--reward-scale",".01","--checkpoint-live-steps","1280","--max-wall-seconds","600",
            "--run-dir",WORK/"smoke-train")
    execute("atc_rl.audit","--run",WORK/"smoke-train","--out",WORK/"smoke-audit.json")
    execute("atc_rl.evaluate","--model",WORK/"smoke-train/model.zip","--episodes","2","--seed","20260","--out",WORK/"smoke-reload")
    summary=json.loads((WORK/"smoke-train/training_summary.json").read_text())
    evaluation=json.loads((WORK/"smoke-reload/protocol.json").read_text())
    assert summary["status"]=="complete" and evaluation["conflict_features"] is True
    verify(SOURCE)
    (WORK/"integration-complete.json").write_text(json.dumps({"parity":record,"smoke_training":summary,
        "reloaded_feature_flag":True,"performance_evidence":False},indent=2),encoding="utf-8")
    print(json.dumps({"integration_complete":True,"smoke_live_transitions":summary["live_transitions"],"performance_evidence":False}),flush=True)
if __name__=="__main__":main()
