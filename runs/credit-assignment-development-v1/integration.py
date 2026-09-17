"""Verify default PPO replay and optional PPO/MAPPO GAE on the real simulator."""
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/onpolicy-credit-source-v1-verify"
sys.path.insert(0,str(SOURCE))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
def call(stage,module,*args):
    print(json.dumps({"stage":stage}),flush=True)
    with (WORK/(stage+".log")).open("x",encoding="utf-8") as f:
        result=subprocess.run([sys.executable,"-u","-m",module,*map(str,args)],cwd=SOURCE,env=os.environ,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError("Failed "+stage+"; all artifacts retained")
def main():
    from atc_rl.cluster import verify
    from atc_rl.checkpoint_identity import policy_fingerprint
    from atc_rl.algorithm_compare import actor_match
    verify(SOURCE)
    old=ROOT/"runs/static-filter-development-v1/ppo/train"
    config=read(old/"config.json")
    fields=("workers","live_steps","rollout_steps","batch_size","epochs","seed","initial_action_std",
            "neutral_action_mean","action_reference","guidance","filter","static_filter","conflict_features",
            "mask_conflict_features","reward_scale","progress_scale","checkpoint_live_steps","max_wall_seconds")
    base=[]
    for key in fields:
        value=config[key]
        if isinstance(value,bool):
            if value:base.append("--"+key.replace("_","-"))
        else:base+=["--"+key.replace("_","-"),str(value)]
    call("default-train","atc_rl.train","--algorithm","ppo",*base,"--run-dir",WORK/"default/train")
    for name in ("initial-model.zip","model.zip"):
        assert policy_fingerprint(old/name)==policy_fingerprint(WORK/"default/train"/name)
    call("default-audit","atc_rl.audit","--run",WORK/"default/train","--out",WORK/"default/audit.json")
    for algorithm in ("ppo","mappo"):
        args=["--algorithm",algorithm,*base,"--gae-lambda",.99,"--run-dir",WORK/algorithm/"train"]
        if algorithm=="mappo":args+=["--actor-reference",WORK/"ppo/train/initial-model.zip"]
        call(algorithm+"-train","atc_rl.train",*args)
        call(algorithm+"-audit","atc_rl.audit","--run",WORK/algorithm/"train","--out",WORK/algorithm/"audit.json")
    assert policy_fingerprint(WORK/"default/train/initial-model.zip")==policy_fingerprint(WORK/"ppo/train/initial-model.zip")
    assert policy_fingerprint(WORK/"default/train/model.zip")!=policy_fingerprint(WORK/"ppo/train/model.zip")
    for algorithm in ("ppo","mappo"):
        assert read(WORK/algorithm/"audit.json")["gae_lambda"]==.99
        assert read(WORK/algorithm/"train/config.json")["traffic_position_scale"]==1
    result={"default_initial_and_final_policy_tensors_unchanged":True,
        "lambda095_and_lambda099_initial_full_policy_tensors_identical":True,
        "lambda099_changed_final_policy_tensors":True,
        "ppo_mappo_initial_actor_match":actor_match(WORK/"ppo/train/initial-model.zip",WORK/"mappo/train/initial-model.zip"),
        "cases":{case:{"summary":read(WORK/case/"train/training_summary.json"),"audit":read(WORK/case/"audit.json")}
                 for case in ("default","ppo","mappo")},"performance_evidence":False}
    verify(SOURCE)
    with (WORK/"integration-complete.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"complete":True,"default_tensor_replay_identical":True,"performance_evidence":False}),flush=True)
if __name__=="__main__":main()
