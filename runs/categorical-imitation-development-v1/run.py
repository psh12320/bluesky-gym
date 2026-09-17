"""Check categorical demonstration fitting and its independently counted PPO handoff."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,subprocess,sys
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    plan=read(OUT/"protocol.json");source=Path(plan["source"])
    sys.path.insert(0,str(source))
    from atc_rl.cluster import verify
    from atc_rl.demonstrations import load_demonstrations,require_disjoint
    from atc_rl.imitation import categorical_labels,verify_teacher_dependencies
    from atc_rl.maneuvers import ManeuverMapping
    from atc_rl.checkpoint_identity import policy_fingerprint
    def preserve():
        verify(source)
        assert sha(plan["bundle"])==plan["bundle_sha256"] and sha(plan["cache"])==plan["cache_sha256"]
        for name,digest in plan["protected_files"].items():assert sha(ROOT/name)==digest,name
    preserve();verify(Path(plan["teacher_source"]))
    (OUT/"started").mkdir(exist_ok=False)
    protocol_sha=sha(OUT/"protocol.json");script_sha=sha(__file__)
    training=load_demonstrations(plan["training_data"],"train")
    validation=load_demonstrations(plan["validation_data"],"validation")
    require_disjoint(training,validation)
    mapping=ManeuverMapping.from_schema(read(Path(plan["training_data"])/"input-schema.json"))
    coverage={name:len(categorical_labels(data,mapping)) for name,data in (("train",training),("validation",validation))}
    dependencies=verify_teacher_dependencies(training,source,plan["teacher_source"])
    with (OUT/"data-preflight.json").open("x",encoding="utf-8") as f:json.dump({"exactly_reconstructed_rows":coverage,"teacher_dependencies":dependencies},f,indent=2)
    environment={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    def execute(module,values,log):
        preserve();assert sha(__file__)==script_sha and sha(OUT/"protocol.json")==protocol_sha
        command=[sys.executable,"-u","-m",module]
        for key,value in values.items():
            flag="--"+key.replace("_","-")
            if isinstance(value,bool):
                if value:command.append(flag)
            else:command.extend([flag,str(value)])
        print(json.dumps({"stage":log.name,"at_utc":datetime.now(timezone.utc).isoformat()}),flush=True)
        with log.open("x",encoding="utf-8") as stream:result=subprocess.run(command,cwd=source,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError("Stage failed; preserve "+str(log))
    execute("atc_rl.imitation",{**plan["supervised"],"training_data":plan["training_data"],"validation_data":plan["validation_data"],
            "teacher_source":plan["teacher_source"],"out":OUT/"bc"},OUT/"bc.log")
    bc=read(OUT/"bc/training_summary.json")
    assert bc["reinforcement_learning_updates"]==0 and bc["epochs"]==1 and bc["supervised_optimizer_steps"]==62
    execute("atc_rl.train",{**plan["rl"],"pretrained_model":OUT/"bc/model.zip","run_dir":OUT/"train"},OUT/"train.log")
    summary=read(OUT/"train/training_summary.json")
    assert summary["status"]=="complete" and 4000<=summary["live_transitions"]<4640
    execute("atc_rl.audit",{"run":OUT/"train","out":OUT/"audit.json"},OUT/"audit.log")
    audit=read(OUT/"audit.json")
    assert not audit["initial_policy_is_untrained"] and audit["pretraining"]["supervised_optimizer_steps"]==62
    assert audit["pretraining"]["ppo_optimizer_inherited"] is False
    assert policy_fingerprint(OUT/"bc/model.zip")==policy_fingerprint(OUT/"train/initial-model.zip")
    preserve()
    result={"status":"complete","finished_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":protocol_sha,
            "script_sha256":script_sha,"exactly_reconstructed_rows":coverage,"supervised":bc,"ppo":summary,"audit":audit,
            "full_initial_tensors_match_supervised_parent":True,"performance_claim":False}
    with (OUT/"complete.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"status":"complete","performance_claim":False,"ppo_live_transitions":summary["live_transitions"]}),flush=True)

if __name__=="__main__":
    try:main()
    except BaseException as error:
        with (OUT/"failure.json").open("x",encoding="utf-8") as f:json.dump({"error":repr(error),"artifacts_retained":True},f,indent=2)
        raise
