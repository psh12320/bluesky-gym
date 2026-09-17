from pathlib import Path
import contextlib,hashlib,importlib.util,io,json,os,shutil,subprocess,sys,zipfile
root=Path.cwd(); work=root/'runs/cluster-ppo-baseline-v1-preparation'
frozen=root/'runs/cluster-ppo-baseline-v1-verify'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((root/'runs/cluster-ppo-baseline-v1.json').read_text())
assert sha(Path(meta['bundle']))==meta['sha256']
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
check=subprocess.run([sys.executable,'-m','atc_rl.cluster','verify'],cwd=frozen,env=env,check=True,capture_output=True,text=True)
with zipfile.ZipFile(root/'runs/onpolicy-learning-source-v2.zip') as base,zipfile.ZipFile(meta['bundle']) as current:
    names=[name for name in base.namelist() if name!='cluster-manifest.json']
    assert len(names)==122
    assert all(base.read(name)==current.read(name) for name in names)
bash=Path(r'C:\Program Files\Git\bin\bash.exe')
for name in ('train_ppo_baseline_v1.slurm','evaluate_ppo_baseline_v1.slurm','submit_ppo_baseline_v1.sh'):
    subprocess.run([str(bash),'-n',str(frozen/'jobs'/name)],check=True)
compile((frozen/'jobs/ppo_baseline_v1.py').read_text(),'ppo_baseline_v1.py','exec')
sbatch_stub=r'''#!/bin/bash
set -euo pipefail
state="$(dirname "$0")/count"
n=0
if [[ -f "$state" ]]; then n=$(cat "$state"); fi
n=$((n + 1))
printf '%s\n' "$n" > "$state"
printf '%s\n' "$*" >> "$(dirname "$0")/calls"
if [[ "${FAIL_SECOND:-0}" == 1 && "$n" == 2 ]]; then exit 4; fi
printf '%s;testcluster\n' "$((930000 + n))"
'''
for mode in ('success','partial-failure'):
    case=work/('shell-'+mode);(case/'jobs').mkdir(parents=True);(case/'bin').mkdir()
    shutil.copyfile(frozen/'jobs/submit_ppo_baseline_v1.sh',case/'jobs/submit_ppo_baseline_v1.sh')
    (case/'bin/sbatch').write_text(sbatch_stub,encoding='utf-8',newline='\n')
    (case/'bin/python3').write_text('#!/bin/bash\nexit 0\n',encoding='utf-8',newline='\n')
    driver='export PATH="$PWD/bin:$PATH"\nexport ATC_PYTHON="$PWD/bin/python3"\nchmod +x bin/*\nbash jobs/submit_ppo_baseline_v1.sh\n'
    (case/'driver.sh').write_text(driver,encoding='utf-8',newline='\n')
    result=subprocess.run([str(bash),'driver.sh'],cwd=case,env={**env,'FAIL_SECOND':'1' if mode=='partial-failure' else '0'},capture_output=True,text=True)
    (case/'validation.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    lines=(case/'bin/calls').read_text().splitlines()
    receipt=case/'runs/cluster-ppo-baseline-v1/submissions.tsv'
    before=receipt.read_bytes()
    if mode=='success':
        assert result.returncode==0 and len(lines)==4
        assert '--array=0' in lines[0] and '--dependency' not in lines[0]
        assert '--array=1-2%1' in lines[1] and 'afterok:930001' in lines[1]
        assert 'afterok:930001' in lines[2] and 'evaluate_ppo_baseline_v1' in lines[2]
        assert 'afterok:930002' in lines[3] and 'evaluate_ppo_baseline_v1' in lines[3]
        assert before.decode().count('\n')==5
    else:
        assert result.returncode!=0 and len(lines)==2 and b'930001' in before and b'930002' not in before
    duplicate=subprocess.run([str(bash),'driver.sh'],cwd=case,env=env,capture_output=True,text=True)
    assert duplicate.returncode!=0 and receipt.read_bytes()==before
    assert (case/'bin/calls').read_text().splitlines()==lines
sys.path.insert(0,str(frozen))
spec=importlib.util.spec_from_file_location('baseline_launch',frozen/'jobs/ppo_baseline_v1.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
plan=json.loads((frozen/'jobs/ppo_baseline_v1.json').read_text())
calls=[]
for mode in ('complete','partial'):
    case=work/('driver-'+mode);(case/'jobs').mkdir(parents=True)
    (case/'jobs/ppo_baseline_v1.json').write_text(json.dumps(plan))
    (case/'cluster-manifest.json').write_text(json.dumps({'files':{}}))
    module.ROOT=case;module.verify=lambda _:None
    def fake_run(name,*args):
        calls.append((mode,name,list(map(str,args))))
        if name=='atc_rl.train':
            argv=list(map(str,args))
            assert '--guidance' not in argv and '--filter' not in argv
            assert '--neutral-action-mean' in argv and argv[argv.index('--device')+1]=='cuda'
            destination=Path(argv[argv.index('--run-dir')+1]);destination.mkdir()
            def dump(name,data):(destination/name).write_text(json.dumps(data))
            dump('config.json',{**plan['config'],'seed':50100})
            dump('training_summary.json',{'status':'complete' if mode=='complete' else 'wall_limit_before_budget',
                                         'live_transitions':1001234 if mode=='complete' else 920000,
                                         'optimizer_steps':999,'model_sha256':'synthetic'})
            dump('provenance.json',{'source_sha256':{}})
        if name=='atc_rl.audit':
            destination=Path(args[1])
            (destination/'checkpoint-audit.json').write_bytes((destination/'training_summary.json').read_bytes())
    module.run=fake_run;module.select_checkpoint=lambda directory,stage:(directory/(stage+'.zip'),{})
    sys.argv=['launcher','train','--index','0']
    with contextlib.redirect_stdout(io.StringIO()):
        if mode=='complete':module.main()
        else:
            try:module.main()
            except ValueError as error:assert 'budget' in str(error)
            else:raise AssertionError('Partial run accepted')
    try:module.main()
    except FileExistsError:pass
    else:raise AssertionError('Existing run overwritten')
    sys.argv=['launcher','evaluate','--index','0'];before_count=len(calls)
    with contextlib.redirect_stdout(io.StringIO()):
        if mode=='complete':module.main()
        else:
            try:module.main()
            except ValueError as error:assert 'budget' in str(error)
            else:raise AssertionError('Partial run evaluated')
    if mode=='complete':
        eval_calls=calls[before_count:]
        assert sum(item[1]=='atc_rl.evaluate' for item in eval_calls)==5
        assert [x[1] for x in eval_calls[-2:]]==['atc_rl.compare','atc_rl.curves']
    else:assert len(calls)==before_count
result={'manifest':json.loads(check.stdout),'bundle_sha256':meta['sha256'],'base_files_unchanged':122,
        'new_python_compiles':True,'bash_syntax_checks':3,
        'simulated_scheduler_checks':['Dependencies and seed arrays','Job IDs retained immediately','Partial submission stops','Duplicate submission preserves receipt'],
        'simulated_driver_checks':['Registered flags','First-seed tests before training','Audit after training','Partial-budget rejection','Duplicate-output rejection','Five evaluations plus comparison and curve'],
        'validation_scope':'Scheduler and driver tests used stubs, not real GPU training. Training/evaluation package bytes are unchanged.',
        'actual_cluster_expanded_suite':'Scheduled in first training allocation; not yet run.','real_cluster_submission':'Not submitted by local validation.'}
(work/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(work/'simulated-driver-calls.json').write_text(json.dumps(calls,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
