"""Prepare the exact Linux interpreter in a separate repository-local runtime."""
import hashlib,json,shutil,subprocess,tarfile,urllib.request
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[2]
plan=json.loads((root/'runs/linux-package-check-v1/runtime-plan.json').read_text())
asset,=plan['matching_python_3_12_14_assets']
assert asset['size']<64*1024**2
assert shutil.disk_usage(root).free>4_000_000_000,'Keep at least 4 GB free before preparation'
out=root/'runs/linux-runtime-v1';out.mkdir(exist_ok=False)
state={'started_at_utc':datetime.now(timezone.utc).isoformat(),'python_asset':asset,
       'scope':'Local WSL interpreter setup; no university connection or simulator evaluation',
       'status':'preparing'}
def save(): (out/'setup-state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
def linux(path):return '/mnt/c/'+path.resolve().as_posix()[3:]
def run(arguments):
    command=['wsl.exe','-d','Ubuntu','--exec',*arguments]
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
    state.setdefault('commands',[]).append({'argv':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    save();result.check_returncode();return result.stdout
save()
try:
    archive=out/asset['name']
    with urllib.request.urlopen(asset['browser_download_url'],timeout=30) as response,archive.open('xb') as stream:
        size=0
        while chunk:=response.read(1024*1024):
            size+=len(chunk);assert size<=asset['size'];stream.write(chunk)
    assert archive.stat().st_size==asset['size']
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert 'sha256:'+digest==asset['digest']
    with tarfile.open(archive) as bundle:
        members=bundle.getmembers()
        assert all(not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
                   and PurePosixPath(m.name).parts[0]=='python' for m in members)
        assert not any(m.isdev() or m.isfifo() for m in members)
    run(['/bin/tar','-xzf',linux(archive),'-C',linux(out)])
    version=run([linux(out/'python/bin/python3.12'),'--version']).strip()
    assert version=='Python 3.12.14',version
    run([linux(out/'python/bin/python3.12'),'-m','venv',linux(out/'.venv')])
    run([linux(out/'.venv/bin/python'),'-m','pip','--version'])
    state.update(status='interpreter_ready',python_version=version,archive_sha256=digest,
                 completed_at_utc=datetime.now(timezone.utc).isoformat(),dependencies_installed=False)
    save();print(json.dumps(state,indent=2))
except BaseException as error:
    state.update(status='failed',error=repr(error));save();raise
