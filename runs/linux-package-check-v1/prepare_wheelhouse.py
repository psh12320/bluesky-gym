"""Fetch Linux wheels through Windows for the network-isolated WSL runtime."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys,urllib.request,zipfile
from packaging.utils import parse_wheel_filename,canonicalize_name
root=Path(__file__).resolve().parents[2];out=root/'runs/linux-runtime-v1'
assert shutil.disk_usage(out).free>4_000_000_000
wheelhouse=out/'wheelhouse';wheelhouse.mkdir(exist_ok=False)
temporary=out/'download-tmp';temporary.mkdir(exist_ok=False)
command=[sys.executable,'-m','pip','download','--disable-pip-version-check','--no-cache-dir',
         '--no-input','--progress-bar','off','--retries','1','--timeout','30','--only-binary=:all:',
         '--python-version','3.12','--implementation','cp','--abi','cp312',
         '--dest',str(wheelhouse),'--constraint',str(out/'constraints-linux.txt'),
         '--requirement',str(out/'requirements-linux.txt')]
for platform in [f'manylinux_2_{minor}_x86_64' for minor in range(35,4,-1)]+['manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64','linux_x86_64']:
    command.extend(['--platform',platform])
state={'started_at_utc':datetime.now(timezone.utc).isoformat(),'status':'downloading','command':command,
       'purpose':'Download Linux packages on Windows; native Linux pip still validates its complete dependency graph offline'}
state_path=out/'wheelhouse-state.json'
def save():state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
    environment=os.environ.copy();environment.update(TEMP=str(temporary),TMP=str(temporary))
    with (out/'wheelhouse-download.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        state['pid']=process.pid;save()
        for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        state['pip_exit_code']=process.wait();save()
        if state['pip_exit_code']:raise RuntimeError('Linux wheel download failed; retain log and partial files')
    plan=json.loads((root/'runs/linux-package-check-v1/runtime-plan.json').read_text())
    files={}
    for path in sorted(wheelhouse.glob('*.whl')):
        name,version,_,tags=parse_wheel_filename(path.name)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if canonicalize_name(name)=='torch':expected=plan['torch_cpu']['sha256'];source=plan['torch_cpu']['url']
        else:
            url=f'https://pypi.org/pypi/{name}/{version}/json'
            with urllib.request.urlopen(url,timeout=30) as response:raw=response.read()
            (wheelhouse/f'{name}-{version}.pypi.json').write_bytes(raw)
            matches=[item for item in json.loads(raw)['urls'] if item['filename']==path.name]
            assert len(matches)==1,path.name
            expected=matches[0]['digests']['sha256'];source=matches[0]['url']
        assert digest==expected,path.name
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            uncompressed=sum(info.file_size for info in archive.infolist())
        files[path.name]={'name':str(name),'version':str(version),'sha256':digest,'source_url':source,
                          'compressed_bytes':path.stat().st_size,'uncompressed_bytes':uncompressed}
    manifest={'files':files,'compressed_bytes':sum(x['compressed_bytes'] for x in files.values()),
              'uncompressed_bytes':sum(x['uncompressed_bytes'] for x in files.values()),
              'all_wheels_match_published_sha256':True,'native_dependency_resolution_pending':True}
    (out/'wheelhouse-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    state.update(status='wheelhouse_ready',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 wheels=len(files),manifest_sha256=hashlib.sha256((out/'wheelhouse-manifest.json').read_bytes()).hexdigest(),
                 disk_free_bytes_after=shutil.disk_usage(out).free)
    save();print(json.dumps({k:v for k,v in state.items() if k!='command'},indent=2))
except BaseException as error:
    state.update(status='failed',error=repr(error));save();raise
