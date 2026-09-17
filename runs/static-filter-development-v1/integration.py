"""Integration checks only; these short runs do not establish performance."""
from pathlib import Path
import csv,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/onpolicy-static-source-v1-verify"
sys.path.insert(0,str(SOURCE))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"

def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def call(label,module,*args):
    print(json.dumps({"stage":label}),flush=True)
    with (WORK/(label+".log")).open("x",encoding="utf-8") as stream:
        result=subprocess.run([sys.executable,"-u","-m",module,*map(str,args)],cwd=SOURCE,
                              stdout=stream,stderr=subprocess.STDOUT,env=os.environ)
    if result.returncode:raise RuntimeError("Stage failed; preserved "+label+".log")
def physical_equal(observed,reference):
    from atc.metrics import METRICS
    def rows(path):
        with path.open(newline="",encoding="utf-8") as stream:
            return {(int(r["episode"]),r["agent"]):r for r in csv.DictReader(stream) if int(r["episode"])<2}
    left,right=rows(observed),rows(reference)
    assert len(left)==len(right)==20
    assert left.keys()==right.keys()
    for key,row in left.items():
        assert row["scenario_sha256"]==right[key]["scenario_sha256"]
        for metric in METRICS:assert float(row[metric])==float(right[key][metric]),(key,metric)
    return len(left)*len(METRICS)

def main():
    import numpy as np
    from atc_rl.cluster import verify
    from atc_rl.world_pool import WorldPool
    from atc_rl.algorithm_compare import actor_match
    verify(SOURCE)
    guards=[]
    for filtered,static,expected in [(False,False,(False,False)),(False,True,(True,False)),
                                     (True,False,(True,True)),(True,True,(True,True))]:
        with_directory=WORK/f"guard-f{int(filtered)}-s{int(static)}"
        env=WorldPool(1,with_directory,guidance=True,filter=filtered,static_filter=static)
        try:
            actual=(env.runtime["static_area_filter"],env.runtime["traffic_conflict_filter"])
            assert actual==expected
            env.seed(20260)
            observation=env.reset()
            next_obs,rewards,dones,infos=env.step(np.zeros((10,2),dtype=np.float32))
            assert np.isfinite(next_obs["actor"]).all() and np.isfinite(rewards).all()
            guards.append({"filter":filtered,"static_filter":static,"runtime":env.runtime})
        finally:env.close()
    (WORK/"guard-selection.json").write_text(json.dumps(guards,indent=2),encoding="utf-8")
    for algorithm in ("ppo","mappo"):
        extra=["--actor-reference",WORK/"ppo/train/initial-model.zip"] if algorithm=="mappo" else []
        call(algorithm+"-train","atc_rl.train","--algorithm",algorithm,"--workers",1,"--live-steps",1280,
             "--rollout-steps",64,"--batch-size",256,"--epochs",2,"--seed",49040,"--device","cpu",
             "--guidance","--static-filter","--conflict-features","--action-reference","goal_offset",
             "--neutral-action-mean","--initial-action-std",.05,"--reward-scale",.01,
             "--checkpoint-live-steps",1280,"--max-wall-seconds",600,
             "--run-dir",WORK/algorithm/"train",*extra)
        call(algorithm+"-audit","atc_rl.audit","--run",WORK/algorithm/"train","--out",WORK/algorithm/"audit.json")
        runtime=read(WORK/algorithm/"train/runtime.json")
        assert runtime["static_area_filter"] and not runtime["traffic_conflict_filter"]
        assert read(WORK/algorithm/"audit.json")["status"]=="complete"
    identity=actor_match(WORK/"ppo/train/initial-model.zip",WORK/"mappo/train/initial-model.zip")
    call("classical-evaluation","atc_rl.evaluate","--classical","--episodes",2,"--seed",20260,"--out",WORK/"classical")
    classical_matches=physical_equal(WORK/"classical/aircraft.csv",ROOT/"runs/ppo-direct-pilot-v1/eval-classical-dev20/aircraft.csv")
    legacy=ROOT/"runs/cpa-mask-development-v1/zeros"
    call("legacy-evaluation","atc_rl.evaluate","--model",legacy/"train/initial-model.zip",
         "--episodes",2,"--seed",20260,"--out",WORK/"legacy")
    legacy_matches=physical_equal(WORK/"legacy/aircraft.csv",legacy/"eval-initial/aircraft.csv")
    runtime=read(WORK/"legacy/summary.json")["runtime"]
    assert not runtime["static_area_filter"] and not runtime["traffic_conflict_filter"]
    for stage,filename in (("initial","initial-model.zip"),("final","model.zip")):
        directory=WORK/"ppo"/("eval-"+stage)
        call("ppo-"+stage+"-evaluation","atc_rl.evaluate","--model",WORK/"ppo/train"/filename,
             "--episodes",2,"--seed",20260,"--out",directory)
        protocol=read(directory/"protocol.json");summary=read(directory/"summary.json")
        assert protocol["static_filter"] and not protocol["filter"]
        assert summary["runtime"]["static_area_filter"] and not summary["runtime"]["traffic_conflict_filter"]
    call("comparison","atc_rl.compare","--initial",WORK/"ppo/eval-initial","--trained",WORK/"ppo/eval-final",
         "--classical",WORK/"classical","--out",WORK/"comparison")
    call("curve","atc_rl.curves","--training-run",WORK/"ppo/train","--classical",WORK/"classical",
         "--evaluation",WORK/"ppo/eval-initial","--evaluation",WORK/"ppo/eval-final","--out",WORK/"curve")
    verify(SOURCE)
    record={"guard_selection_cases":4,"classical_physical_metrics_unchanged":classical_matches,
            "legacy_checkpoint_physical_metrics_unchanged":legacy_matches,
            "ppo_mappo_initial_actor_identity":identity,"smoke_live_transitions":{
                a:read(WORK/a/"train/training_summary.json")["live_transitions"] for a in ("ppo","mappo")},
            "static_only_checkpoint_reload_verified":True,"performance_evidence":False}
    (WORK/"integration-complete.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record),flush=True)
if __name__=="__main__":main()
