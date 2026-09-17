"""Run both archived development checks using the separate native Linux runtime."""
import hashlib,json,os,subprocess,zipfile
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[2];runtime=root/'runs/linux-runtime-v1'
installation=json.loads((runtime/'install-success.json').read_text())
assert installation['status']=='dependencies_ready'
assert hashlib.sha256((runtime/installation['audit_file']).read_bytes()).hexdigest()==installation['audit_sha256']
assert hashlib.sha256((runtime/'packages.txt').read_bytes()).hexdigest()==installation['installed_packages_sha256']
archive=root/'output/candidates/decimal-heading-v1.zip'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(archive)=='da020cf4079af110fcfcb7598b13f58e84859f8b5a0e549d9a96ad9ab5c86b2c'
out=root/'runs/linux-reproduction-v1';out.mkdir(exist_ok=False)
def linux(path):return '/mnt/c/'+path.resolve().as_posix()[3:]
python=linux(runtime/'.venv/bin/python')
prefix=['wsl.exe','-d','Ubuntu','--exec','/usr/bin/env','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'OPENBLAS_NUM_THREADS=1','PYTHONUNBUFFERED=1','SDL_VIDEODRIVER=dummy','PYGAME_HIDE_SUPPORT_PROMPT=1',
        'TMPDIR='+linux(runtime/'tmp'),python]
protocol={'started_at_utc':datetime.now(timezone.utc).isoformat(),'candidate_archive_sha256':sha(archive),
          'python_setup_sha256':sha(runtime/'setup-state.json'),'installation_state_sha256':sha(runtime/'install-success.json'),
          'installed_packages_sha256':sha(runtime/'packages.txt'),'tracks':['sa','ma'],'seed':2026,'scenarios_per_track':2,
          'scope':'Native Linux reproduction of fixed development prefixes; not new unseen performance evidence or a university job',
          'pass_rule':'Archived check requires exact physical metrics and reward error at most 1e-5; retain every failure'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
state={'status':'running','protocol_sha256':sha(out/'protocol.json'),'stages':[]}
def save():(out/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
def run(track,name,arguments):
    command=prefix+arguments;item={'track':track,'name':name,'command':command,'started_at_utc':datetime.now(timezone.utc).isoformat()}
    state['stages'].append(item);save()
    with (out/f'{track}-{name}.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        item['pid']=process.pid;save()
        for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        item.update(exit_code=process.wait(),finished_at_utc=datetime.now(timezone.utc).isoformat());save()
    return item['exit_code']==0
save();results={}
for track in protocol['tracks']:
    candidate=out/track;candidate.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        bundle.extractall(candidate)
    verified=run(track,'verify-before',[linux(candidate/'verify_contents.py')])
    if not verified:
        results[track]={'passed':False,'stage':'verify-before'};continue
    passed=run(track,'development',[linux(candidate/'check_development.py'),'--env',track])
    verified_after=run(track,'verify-after',[linux(candidate/'verify_contents.py')])
    results[track]={'passed':passed and verified_after,'source_verified_after':verified_after}
    if passed:
        check=candidate/f'checks/{track}.json';results[track].update(check=json.loads(check.read_text()),check_sha256=sha(check))
state.update(status='complete' if all(r['passed'] for r in results.values()) else 'failed',results=results,
             completed_at_utc=datetime.now(timezone.utc).isoformat())
save();print(json.dumps(state,indent=2))
if state['status']!='complete':raise SystemExit('One or more native Linux checks failed; preserve every log and investigate')
