"""Install pinned evaluation dependencies in the repository-local WSL environment."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
root=Path(__file__).resolve().parents[2];out=root/'runs/linux-runtime-v1'
setup=json.loads((out/'setup-state.json').read_text())
assert setup['status']=='interpreter_ready'
assert shutil.disk_usage(root).free>4_000_000_000,'Keep at least 4 GB free before dependency installation'
state_path=out/'install-state.json';assert not state_path.exists(),'Preserve the previous installation attempt'
source=root/'runs/decimal-transport-candidates-v1/package-source/packages.txt'
constraints=[line for line in source.read_text().splitlines() if '==' in line and not line.startswith('#')]
(out/'constraints-linux.txt').write_text('\n'.join(constraints)+'\n',encoding='utf-8')
plan=json.loads((root/'runs/linux-package-check-v1/runtime-plan.json').read_text())
wheel=plan['torch_cpu'];assert wheel['gpu_dependencies_declared'] is False
requirements=['torch @ '+wheel['url']+'#sha256='+wheel['sha256'],
              'bluesky-simulator','gymnasium','numpy','pettingzoo','pygame','PyQt6','shapely','SuperSuit','stable-baselines3']
(out/'requirements-linux.txt').write_text('\n'.join(requirements)+'\n',encoding='utf-8')
def linux(path):return '/mnt/c/'+path.resolve().as_posix()[3:]
python=linux(out/'.venv/bin/python')
(out/'tmp').mkdir(exist_ok=True)
command=['wsl.exe','-d','Ubuntu','--exec','/usr/bin/env','TMPDIR='+linux(out/'tmp'),python,'-m','pip','install','--disable-pip-version-check',
         '--no-cache-dir','--only-binary=:all:','--report',linux(out/'installation-report.json'),
         '--constraint',linux(out/'constraints-linux.txt'),'--requirement',linux(out/'requirements-linux.txt')]
state={'started_at_utc':datetime.now(timezone.utc).isoformat(),'status':'installing','command':command,
       'source_packages_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'source_constraints':'Frozen Windows package versions, excluding editable local project path',
       'torch_cpu_wheel_sha256':wheel['sha256'],'simulator_evaluation_started':False}
def save():state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
    with (out/'install.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        state['pid']=process.pid;save()
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        status=process.wait();state['pip_exit_code']=status;save()
        if status:raise RuntimeError(f'pip exited with {status}; preserve install.log')
    check=subprocess.run(['wsl.exe','-d','Ubuntu','--exec',python,'-m','pip','check'],capture_output=True,text=True)
    (out/'pip-check.txt').write_text(check.stdout+check.stderr,encoding='utf-8');check.check_returncode()
    freeze=subprocess.run(['wsl.exe','-d','Ubuntu','--exec',python,'-m','pip','freeze'],capture_output=True,text=True)
    freeze.check_returncode();(out/'packages.txt').write_text(freeze.stdout,encoding='utf-8')
    state.update(status='dependencies_ready',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 installed_packages_sha256=hashlib.sha256((out/'packages.txt').read_bytes()).hexdigest(),
                 disk_free_bytes_after=shutil.disk_usage(root).free)
    save();print(json.dumps(state,indent=2))
except BaseException as error:
    state.update(status='failed',error=repr(error));save();raise
