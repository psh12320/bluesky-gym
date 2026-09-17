
"""Exercise gSDE in the actual simulator; these short runs are not performance evidence."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/onpolicy-gsde-source-v1-verify"
sys.path.insert(0,str(SOURCE))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))

def call(name,module,*args):
    print(json.dumps({"stage":name}),flush=True)
    with (WORK/(name+".log")).open("x",encoding="utf-8") as log:
        result=subprocess.run([sys.executable,"-u","-m",module,*map(str,args)],cwd=SOURCE,
            env=os.environ,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError("Failed "+name+"; logs preserved")

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
    call("gaussian-train","atc_rl.train","--algorithm","ppo",*base,"--run-dir",WORK/"gaussian/train")
    identical={}
    for name in ("initial-model.zip","model.zip"):
        identical[name]=policy_fingerprint(old/name)==policy_fingerprint(WORK/"gaussian/train"/name)
        assert identical[name],"Default Gaussian training changed "+name
    for algorithm in ("ppo","mappo"):
        folder=WORK/algorithm
        args=["--algorithm",algorithm,*base,"--exploration","gsde","--sde-weight-std",.05,
              "--sde-sample-freq",12,"--run-dir",folder/"train"]
        if algorithm=="mappo":
            args+=["--actor-reference",WORK/"ppo/train/initial-model.zip"]
        call(algorithm+"-train","atc_rl.train",*args)
        call(algorithm+"-audit","atc_rl.audit","--run",folder/"train","--out",folder/"audit.json")
        call(algorithm+"-native","atc_rl.competition","--env","ma","--model",folder/"train/model.zip",
             "--episodes",2,"--seed",20260,"--out",folder/"native")
    identity=actor_match(WORK/"ppo/train/initial-model.zip",WORK/"mappo/train/initial-model.zip")
    result={"gaussian_initial_and_trained_policy_tensors_identical":identical,
            "gsde_initial_actor_match":identity,"performance_evidence":False,"training":{}}
    for algorithm in ("ppo","mappo"):
        folder=WORK/algorithm
        audit=read(folder/"audit.json")
        evaluation=read(folder/"native/summary.json")
        protocol=read(folder/"native/protocol.json")
        assert audit["exploration"]=="gsde" and audit["sde_sample_freq"]==12
        assert evaluation["episodes"]==2 and evaluation["agent_episodes"]==20
        assert protocol["exploration"]=="gsde" and protocol["sde_sample_freq"]==12
        assert audit["actor_gradient_to_joint_context_zero"]
        assert audit["critic_uses_joint_context"]==(algorithm=="mappo")
        result["training"][algorithm]={"summary":read(folder/"train/training_summary.json"),
            "audit":audit,"native_evaluation_csv_sha256":evaluation["csv_sha256"]}
    verify(SOURCE)
    with (WORK/"integration-complete.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"complete":True,"gaussian_policy_replay_identical":True,
                     "gsde_algorithms":["ppo","mappo"],"performance_evidence":False}),flush=True)

if __name__=="__main__":main()
