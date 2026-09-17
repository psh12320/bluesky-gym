"""Profile the fixed MA deployment on its archived two-scenario development check."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,pstats,subprocess,sys,time,zipfile
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
archive=root/'output/candidates/decimal-heading-v1.zip'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(archive)=='da020cf4079af110fcfcb7598b13f58e84859f8b5a0e549d9a96ad9ab5c86b2c'
candidate=out/'candidate';candidate.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as bundle:assert bundle.testzip() is None;bundle.extractall(candidate)
command=[sys.executable,'-u','-m','cProfile','-o',str(out/'profile.pstats'),str(candidate/'check_development.py'),'--env','ma']
protocol={'started_at_utc':datetime.now(timezone.utc).isoformat(),'candidate_archive_sha256':sha(archive),
          'track':'ma','seed':2026,'episodes':2,'command':command,
          'scope':'Diagnostic profile of a fixed development prefix; no policy changes or new performance-selection data',
          'timing_limitations':'cProfile overhead and other simultaneous local jobs; not an isolated throughput benchmark'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
state={'status':'running','protocol_sha256':sha(out/'protocol.json')}
def save():(out/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ.update(SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
save();start=time.perf_counter()
try:
    with (out/'profile.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        state['pid']=process.pid;save()
        for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        state['exit_code']=process.wait();save()
        if state['exit_code']:raise RuntimeError('Profiled development check failed; preserve output')
    check=json.loads((candidate/'checks/ma.json').read_text());assert check['all_metrics_match']
    manifest=json.loads((candidate/'manifest.json').read_text())
    for name,digest in manifest['files'].items():assert sha(candidate/name)==digest,name
    stats=pstats.Stats(str(out/'profile.pstats'))
    rows=[{'file':key[0],'line':key[1],'function':key[2],'primitive_calls':v[0],'total_calls':v[1],
           'self_seconds':v[2],'cumulative_seconds':v[3]} for key,v in stats.stats.items()]
    result={'wall_seconds':time.perf_counter()-start,'profiler_total_seconds':stats.total_tt,
            'total_calls':stats.total_calls,'primitive_calls':stats.prim_calls,
            'top_self_time':sorted(rows,key=lambda x:x['self_seconds'],reverse=True)[:40],
            'top_controller_cumulative_time':sorted([r for r in rows if '/atc/' in r['file'].replace(chr(92),'/')],key=lambda x:x['cumulative_seconds'],reverse=True)[:40],
            'all_metric_checks_passed':True,'unchanged_packaged_files':len(manifest['files']),
            'scope':protocol['scope'],'timing_limitations':protocol['timing_limitations']}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    state.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 summary_sha256=sha(out/'summary.json'),profile_sha256=sha(out/'profile.pstats'),development_check_sha256=sha(candidate/'checks/ma.json'))
    save();print(json.dumps({'status':'complete','wall_seconds':result['wall_seconds'],'all_metric_checks_passed':True},indent=2))
except BaseException as error:
    state.update(status='failed',error=repr(error));save();raise
