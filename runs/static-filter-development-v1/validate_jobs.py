"""Exercise every job row without Slurm or simulator work."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/cluster-ppo-static-v1-verify'
sys.path.insert(0,str(SOURCE))
from atc_rl.cluster import verify
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    verify(SOURCE)
    plan=read(SOURCE/'jobs/ppo_static_v1.json')
    assert len(plan['rows'])==6
    assert {(r['seed'],r['arm']) for r in plan['rows']}=={(s,a) for s in plan['seeds'] for a in plan['arms']}
    manifest=read(SOURCE/'cluster-manifest.json')['files']
    for name in manifest:
        if name.endswith('.py'):ast.parse((SOURCE/name).read_text(encoding='utf-8'),filename=name)
    base=read(ROOT/'runs/onpolicy-static-source-v1-verify/cluster-manifest.json')['files']
    assert all(manifest[k]==v for k,v in base.items())
    previous=read(ROOT/'runs/onpolicy-cpa-source-v2-verify/cluster-manifest.json')['files']
    fixed={k:v for k,v in previous.items() if k.split('/')[0] in ('atc','core','bluesky_gym','bluesky_zoo')}
    assert all(manifest[k]==v for k,v in fixed.items())
    archive_checks={}
    for name,expected in read(ROOT/'runs/cpa-mask-development-v1/validation.json')['verified_archive_sha256'].items():
        observed=hashlib.sha256((ROOT/'runs'/f'{name}.zip').read_bytes()).hexdigest()
        assert observed==expected
        archive_checks[name]=observed
    checks=[]
    spec=importlib.util.spec_from_file_location('static_job',SOURCE/'jobs/ppo_static_v1.py')
    job=importlib.util.module_from_spec(spec);spec.loader.exec_module(job)
    source={k:v for k,v in manifest.items() if k=='pyproject.toml' or k.endswith('.py') and k.split('/')[0] in ('atc','atc_rl','core','bluesky_gym','bluesky_zoo')}
    for index,row in enumerate(plan['rows']):
        sandbox=WORK/'job-dry-run'/str(index);(sandbox/'jobs').mkdir(parents=True)
        (sandbox/'jobs/ppo_static_v1.json').write_text(json.dumps(plan))
        (sandbox/'cluster-manifest.json').write_text(json.dumps({'files':manifest}))
        training=sandbox/'runs/cluster-ppo-static-v1'/f'seed-{row["seed"]}'/row['arm']/'train'
        expected={**plan['config'],**plan['arms'][row['arm']],'seed':row['seed']}
        commands=[]
        def run(module,*args):
            commands.append([module,*map(str,args)])
            if module=='atc_rl.train':
                argv=list(map(str,args))
                assert '--static-filter' in argv and '--guidance' in argv and '--conflict-features' in argv
                assert '--filter' not in argv
                assert ('--mask-conflict-features' in argv)==(row['arm']=='zeros')
                assert argv[argv.index('--seed')+1]==str(row['seed'])
                training.mkdir()
                (training/'config.json').write_text(json.dumps(expected))
                summary={'status':'complete','live_transitions':1001000,'optimizer_steps':100,'model_sha256':'fixture-only'}
                (training/'training_summary.json').write_text(json.dumps(summary))
                (training/'checkpoint-audit.json').write_text(json.dumps(summary))
                (training/'runtime.json').write_text(json.dumps({'static_area_filter':True,'traffic_conflict_filter':False}))
                (training/'provenance.json').write_text(json.dumps({'source_sha256':source}))
        job.ROOT=sandbox;job.verify=lambda p:None;job.run=run
        sys.argv=['ppo_static_v1.py','train','--index',str(index)]
        job.main()
        job.select_checkpoint=lambda folder,stage:(folder/(stage+'.zip'),{})
        sys.argv=['ppo_static_v1.py','evaluate','--index',str(index)]
        job.main()
        evaluated=[c for c in commands if c[0]=='atc_rl.evaluate']
        assert len(evaluated)==5 and '--classical' in evaluated[0]
        assert sum(c[0]=='pytest' for c in commands)==int(index==0)
        checks.append({'index':index,**row,'command_count':len(commands),'evaluation_count':len(evaluated)})
    result={'six_job_rows_dry_run':checks,'all_cluster_base_files_identical':len(base),
            'unchanged_controller_and_simulator_files':len(fixed),'preserved_old_archives':archive_checks,
            'slurm_syntax_checked_separately':True,'simulator_integration':'See integration-complete.json',
            'performance_jobs_submitted':False}
    (WORK/'job-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'validated_job_rows':len(checks),'unchanged_controller_files':len(fixed)}))
if __name__=='__main__':main()
