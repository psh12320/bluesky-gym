"""Resolve and install the verified wheelhouse offline with native Linux pip."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
root=Path(__file__).resolve().parents[2];out=root/'runs/linux-runtime-v1'
read=lambda path:json.loads(path.read_text())
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
original=read(out/'install-state.json')
assert original['status']=='failed','Do not run two package installers in one environment'
assert read(out/'wheelhouse-v2-state.json')['status']=='wheelhouse_ready'
manifest=read(out/'wheelhouse-v2-manifest.json')
for name,item in manifest['files'].items():assert sha(out/'wheelhouse-v2'/name)==item['sha256'],name
assert shutil.disk_usage(out).free>manifest['uncompressed_bytes']+1_500_000_000
state_path=out/'offline-install-state.json';assert not state_path.exists(),'Preserve the previous offline installation attempt'
requirements=(out/'requirements-linux.txt').read_text().splitlines()
assert requirements[0].startswith('torch @ ')
requirements[0]='torch==2.14.0+cpu'
(out/'requirements-offline.txt').write_text('\n'.join(requirements)+'\n',encoding='utf-8')
def linux(path):return '/mnt/c/'+path.resolve().as_posix()[3:]
python=linux(out/'.venv/bin/python')
prefix=['wsl.exe','-d','Ubuntu','--exec','/usr/bin/env','TMPDIR='+linux(out/'tmp'),python]
command=prefix+['-m','pip','install','--disable-pip-version-check','--no-cache-dir','--no-index',
                '--only-binary=:all:','--find-links',linux(out/'wheelhouse-v2'),
                '--report',linux(out/'offline-installation-report.json'),
                '--constraint',linux(out/'constraints-linux.txt'),'--requirement',linux(out/'requirements-offline.txt')]
state={'started_at_utc':datetime.now(timezone.utc).isoformat(),'status':'installing','command':command,
       'original_failed_attempt_sha256':sha(out/'install-state.json'),
       'wheelhouse_manifest_sha256':sha(out/'wheelhouse-v2-manifest.json'),
       'native_linux_dependency_resolution':True,'network_access':False,'simulator_evaluation_started':False}
def save():state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
    with (out/'offline-install.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        state['pid']=process.pid;save()
        for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        state['pip_exit_code']=process.wait();save()
        if state['pip_exit_code']:raise RuntimeError('Offline dependency resolution or installation failed; preserve log')
    for name,args in [('pip-check',['-m','pip','check']),('packages',['-m','pip','freeze'])]:
        result=subprocess.run(prefix+args,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (out/f'{name}.txt').write_text(result.stdout+result.stderr,encoding='utf-8');result.check_returncode()
    state.update(status='dependencies_ready',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 installed_packages_sha256=sha(out/'packages.txt'),disk_free_bytes_after=shutil.disk_usage(out).free)
    save()
    success=out/'install-success.json';assert not success.exists()
    success.write_text(json.dumps({'status':'dependencies_ready','method':'offline',
                      'audit_file':state_path.name,'audit_sha256':sha(state_path),
                      'installed_packages_sha256':sha(out/'packages.txt')},indent=2),encoding='utf-8')
    print(json.dumps(state,indent=2))
except BaseException as error:
    state.update(status='failed',error=repr(error));save();raise
