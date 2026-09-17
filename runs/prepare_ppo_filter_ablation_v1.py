from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
root=Path.cwd();parent=root/'runs/ppo-filter-ablation-v1'
parent.mkdir(exist_ok=False)
source=root/'runs/onpolicy-learning-source-v2-verify'
evaluation=root/'runs/onpolicy-goal-offset-source-v1-verify'
for directory in (source,evaluation):
    manifest=json.loads((directory/'cluster-manifest.json').read_text())
    assert all(hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest for name,digest in manifest['files'].items())
protocol={'registered_at_utc':datetime.now(timezone.utc).isoformat(),
 'purpose':'Measure learned PPO changes with conflict filtering enabled, with route guidance separately off or on.',
 'seeds':[49900,49920,49940],'support_groups':[{'name':'g0-f1','guidance':False,'filter':True},{'name':'g1-f1','guidance':True,'filter':True}],
 'world_seeds':{str(seed):[seed,seed+10] for seed in (49900,49920,49940)},
 'config':{'algorithm':'ppo','workers':2,'live_steps':100000,'rollout_steps':256,'batch_size':1024,'epochs':10,
           'device':'cpu','initial_action_std':.05,'neutral_action_mean':True,'action_reference':'goal_offset',
           'reward_scale':.01,'progress_scale':0.,'checkpoint_live_steps':50000,'max_wall_seconds':3600},
 'training_source':str(source),'training_bundle_sha256':'33579f122d1fffd6bf6f9aefdabad846111afe20cbaadb56e390ae2d5a849b8c',
 'evaluation_source':str(evaluation),'evaluation_seed':20260,'evaluation_worlds':20,
 'classical_evaluation':str(root/'runs/ppo-direct-pilot-v1/eval-classical-dev20'),
 'guidance_only_comparison':str(root/'runs/ppo-guidance-replication-v1/protocol.json'),
 'primary_learning_measure':'Per-seed final-minus-own-initial change in all nine native metrics, clean completion and all-aircraft-clean completion.',
 'aggregation':'Keep all three seed effects; report their mean and sample standard deviation. Paired-world bootstrap intervals describe scenarios, not training-seed uncertainty.',
 'support_comparison':'Guidance effect with filtering fixed on; filter effect with guidance fixed on using the completed guidance-only cohort. Compare changes from own initial policies, not just final scores.',
 'advancement_policy':'No automatic selection or longer-budget launch. Review learned physical changes across all seeds; high initial support scores do not count as learning.',
 'limitations':['100k feasibility pilots, not the canonical eight-world million-transition matrix.',
                'Goal-offset actions supply a navigation prior even with guidance disabled.',
                'These six runs and the existing three guidance-only runs do not constitute a complete four-cell, three-seed training matrix.',
                'Previously used development worlds; unseen streams 20301/20302 remain reserved.',
                'The historical +5-point clean-completion gate is not used to rank a control already at 97%; no previous screen outcome is changed.'],
 'scheduling':'Two coordinators, one per support group; each trains two worlds and evaluates one world sequentially per seed. Existing evaluation continues.',
 'failure_behavior':'Stop that coordinator on errors or incomplete budgets, retain all outputs and do not relaunch automatically.'}
(parent/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
driver=r'''"""Run one precommitted filter-ablation cohort without changing frozen code."""
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
'''
compile(driver,'run.py','exec')
(parent/'run.py').write_text(driver,encoding='utf-8')
print(json.dumps({'protocol':str(parent/'protocol.json'),'training_started':False,'registered_training_runs':6,'existing_sources_verified':True}))
