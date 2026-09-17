"""Exercise registered categorical PPO/MAPPO training and native deployment."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    plan=read(OUT/"protocol.json")
    source=Path(plan["source"])
    sys.path.insert(0,str(source))
    from atc_rl.cluster import verify
    from atc.metrics import METRICS
    def preserve():
        verify(source)
        assert sha(plan["bundle"])==plan["bundle_sha256"]
        assert sha(plan["cache"])==plan["cache_sha256"]
        for name,digest in plan["protected_files"].items():
            assert sha(ROOT/name)==digest,name
    preserve()
    (OUT/"started").mkdir(exist_ok=False)
    protocol_sha=sha(OUT/"protocol.json")
    script_sha=sha(__file__)
    environment={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    def execute(module,arguments,log):
        preserve()
        assert sha(__file__)==script_sha and sha(OUT/"protocol.json")==protocol_sha
        command=[sys.executable,"-u","-m",module]
        for key,value in arguments.items():
            if isinstance(value,bool):
                if value:command.append("--"+key.replace("_","-"))
            else:command.extend(["--"+key.replace("_","-"),str(value)])
        print(json.dumps({"stage":str(log.relative_to(OUT)),"at_utc":datetime.now(timezone.utc).isoformat()}),flush=True)
        with log.open("x",encoding="utf-8") as stream:
            result=subprocess.run(command,cwd=source,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError("Stage failed; preserve "+str(log))
    def compare(left,right):
        def rows(directory):
            with (directory/"aircraft.csv").open(newline="",encoding="utf-8") as stream:
                data=list(csv.DictReader(stream))
            index={(r["episode"],r["scenario_sha256"],r["agent"]):r for r in data}
            assert len(index)==len(data)==20
            return index
        a,b=rows(left),rows(right)
        assert a.keys()==b.keys(),"Scenario or aircraft identity mismatch"
        for key in a:
            for metric in METRICS:
                if float(a[key][metric])!=float(b[key][metric]):
                    raise ValueError(f"Metric mismatch at {key}, {metric}")
        return len(a)*len(METRICS)
    results={}
    for algorithm in plan["arms"]:
        folder=OUT/algorithm
        folder.mkdir(exist_ok=False)
        config={**plan["config"],"algorithm":algorithm,"run_dir":folder/"train"}
        if algorithm=="mappo":config["actor_reference"]=OUT/plan["mappo_actor_reference"]
        execute("atc_rl.train",config,folder/"train.log")
        summary=read(folder/"train/training_summary.json")
        assert summary["status"]=="complete"
        assert plan["config"]["live_steps"]<=summary["live_transitions"]<plan["config"]["live_steps"]+640
        execute("atc_rl.audit",{"run":folder/"train","out":folder/"audit.json"},folder/"audit.log")
        checked=read(folder/"audit.json")
        assert checked["actor_gradient_to_joint_context_zero"]
        assert checked["critic_uses_joint_context"]==(algorithm=="mappo")
        with (folder/"train/learning.csv").open(newline="",encoding="utf-8") as stream:
            learning=list(csv.DictReader(stream))
        assert all(row["rollout_action_box_clip_fraction_live"]=="" for row in learning)
        for stage,filename in (("initial","initial-model.zip"),("final","model.zip")):
            execute("atc_rl.evaluate",{"model":folder/"train"/filename,**plan["evaluation"],"out":folder/f"eval-{stage}"},folder/f"eval-{stage}.log")
        execute("atc_rl.competition",{"model":folder/"train/model.zip",**plan["evaluation"],"env":"ma","out":folder/"native"},folder/"native.log")
        matched=compare(folder/"eval-final",folder/"native")
        assert read(folder/"native/summary.json")["runtime"]["traffic_conflict_filter"] is False
        results[algorithm]={"training":summary,"audit":checked,"matched_native_vector_metric_values":matched,
                            "initial":read(folder/"eval-initial/summary.json"),"final":read(folder/"eval-final/summary.json")}
        with (folder/"complete.json").open("x",encoding="utf-8") as f:json.dump(results[algorithm],f,indent=2)
        print(json.dumps({"algorithm_complete":algorithm,"live_transitions":summary["live_transitions"],"matched_metrics":matched}),flush=True)
    from stable_baselines3 import PPO
    import torch
    from atc_rl.actor_reference import is_actor_parameter
    first=PPO.load(OUT/"ppo/train/initial-model.zip",device="cpu")
    second=PPO.load(OUT/"mappo/train/initial-model.zip",device="cpu")
    left=dict(first.policy.named_parameters());right=dict(second.policy.named_parameters())
    names=[name for name in left if is_actor_parameter(name)]
    assert all(torch.equal(left[name],right[name]) for name in names)
    assert read(OUT/"mappo/train/actor-reference/matching.json")["actor_parameters_identical"]
    initial_matches=compare(OUT/"ppo/eval-initial",OUT/"mappo/eval-initial")
    preserve()
    result={"status":"complete","finished_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":protocol_sha,
            "script_sha256":script_sha,"arms":results,"matched_initial_actor_tensors":len(names),
            "matched_initial_metric_values":initial_matches,"performance_claim":False,
            "scope":"Real-simulator training, active masking, serialization, information separation and native deployment checks. The budget and scenario count do not establish competitive performance."}
    with (OUT/"complete.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"status":"complete","performance_claim":False,"algorithms":list(results)}),flush=True)


if __name__=="__main__":
    try:main()
    except BaseException as error:
        with (OUT/"failure.json").open("x",encoding="utf-8") as f:json.dump({"error":repr(error),"artifacts_retained":True},f,indent=2)
        raise
