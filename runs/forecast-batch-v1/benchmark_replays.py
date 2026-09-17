"""Run declared, interleaved development timing checks without changing the release."""
import argparse,contextlib,hashlib,json,os,runpy,shutil,subprocess,sys,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ.update(SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
read=lambda path:json.loads(path.read_text())
ARCHIVE=ROOT/'output/candidates/decimal-heading-v1.zip'
CACHE=ROOT/'runs/simulator/cache/navdata.p'

def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def child(mode,ordinal):
    out=BASE/f'timing-ma/{ordinal:02d}-{mode}';out.mkdir(exist_ok=False)
    candidate=out/'candidate';candidate.mkdir()
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None;archive.extractall(candidate)
    source=candidate/'source';cache=source/'runs/simulator/cache/navdata.p'
    cache.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(CACHE,cache)
    assert sha(CACHE)==sha(cache)
    sys.path.insert(0,str(source))
    stats={'mode':mode,'decision_batches':0,'aircraft_forecasts':0,'actions':0,'command_seconds':0.0,'first_command_time':None,'last_command_time':None}
    wall_start=time.perf_counter()
    from atc import deployment
    original_make_env=deployment.make_env
    def make_env(*args,**kwargs):
        env=original_make_env(*args,**kwargs)
        import bluesky as bs
        import numpy as np
        from atc.traffic_projection import predict_commands,capture_times
        raw=env.unwrapped;original_action=raw._get_action
        def action(ac_id,value):
            began=time.perf_counter()
            if stats['first_command_time'] is None:stats['first_command_time']=began
            try:
                if mode=='batch' and raw._traffic_plan_time!=float(bs.sim.simt):
                    agents=list(bs.traf.id);count=len(agents);assert count>0
                    positions=np.stack([raw._projection_xy((bs.traf.lat[i],bs.traf.lon[i])) for i in range(count)])
                    headings=np.asarray(bs.traf.hdg,dtype=float).copy();speeds=np.asarray(bs.traf.tas,dtype=float).copy()
                    paths=predict_commands(positions,headings,speeds,np.zeros(count),speeds)
                    plans={}
                    for i,agent in enumerate(agents):
                        goal=raw._goal.get(agent) if hasattr(raw,'_goal') else None
                        target=raw._projection_xy(goal) if goal is not None else None
                        life=capture_times(paths[i:i+1],target,raw.distance_margin)[0]
                        plans[agent]=(paths[i].copy(),life)
                    raw._traffic_plans=plans;raw._traffic_plan_time=float(bs.sim.simt)
                    stats['decision_batches']+=1;stats['aircraft_forecasts']+=count
                return original_action(ac_id,value)
            finally:
                ended=time.perf_counter();stats['command_seconds']+=ended-began
                stats['actions']+=1;stats['last_command_time']=ended
        raw._get_action=action
        return env
    deployment.make_env=make_env
    sys.argv=[str(candidate/'check_development.py'),'--env','ma']
    runpy.run_path(str(candidate/'check_development.py'),run_name='__main__')
    check=read(candidate/'checks/ma.json');assert check['all_metrics_match']
    assert all(v==0 for v in check['metric_max_absolute_difference'].values())
    manifest=read(candidate/'manifest.json')
    for name,digest in manifest['files'].items():assert sha(candidate/name)==digest,name
    result={'mode':mode,'ordinal':ordinal,'wall_seconds':time.perf_counter()-wall_start,
            'active_replay_seconds':stats.pop('last_command_time')-stats.pop('first_command_time'),
            'command_statistics':stats,'development_check':check,'unchanged_packaged_files':len(manifest['files']),
            'candidate_archive_sha256':sha(ARCHIVE),'navigation_cache_sha256':sha(CACHE),'per_forecast_scalar_verification':False}
    dump(out/'summary.json',result)

def parent():
    import statistics
    for track in ('ma','sa'):
        previous=BASE/f'replay-{track}-retry2';state=read(previous/'state.json');result=read(previous/'summary.json')
        assert state['status']=='complete' and sha(previous/'summary.json')==state['summary_sha256']
        assert result['development_check']['all_metrics_match']
        assert result['prediction_checks']['all_forecast_coordinates_byte_equal'] and result['prediction_checks']['all_lifetimes_byte_equal']
    assert sha(ARCHIVE)=='da020cf4079af110fcfcb7598b13f58e84859f8b5a0e549d9a96ad9ab5c86b2c'
    out=BASE/'timing-ma';out.mkdir(exist_ok=False)
    protocol={'started_at_utc':datetime.now(timezone.utc).isoformat(),'track':'ma','seed':2026,'scenarios_per_run':2,
              'order':['scalar','batch','batch','scalar'],'pairs':[[0,1],[3,2]],'script_sha256':sha(Path(__file__)),
              'candidate_archive_sha256':sha(ARCHIVE),'navigation_cache_sha256':sha(CACHE),
              'prior_exact_replays':{t:sha(BASE/f'replay-{t}-retry2/summary.json') for t in ('ma','sa')},
              'pass_rule':'All nine archived development metrics must match exactly for every run; all package files must remain unchanged',
              'scope':'Descriptive local timings from two ordered pairs under concurrent scoring load. No confidence interval, cluster throughput or universal speedup claim. No model selection.',
              'scalar_recheck_disabled_for_timing':True,'production_source_changed':False}
    dump(out/'protocol.json',protocol)
    state={'status':'running','protocol_sha256':sha(out/'protocol.json'),'stages':[]};dump(out/'state.json',state)
    try:
        for ordinal,mode in enumerate(protocol['order']):
            stage={'ordinal':ordinal,'mode':mode,'started_at_utc':datetime.now(timezone.utc).isoformat()};state['stages'].append(stage)
            with (out/f'{ordinal:02d}-{mode}.log').open('x',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--mode',mode,'--ordinal',str(ordinal)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                stage['pid']=process.pid;dump(out/'state.json',state);stage['exit_code']=process.wait()
            stage['finished_at_utc']=datetime.now(timezone.utc).isoformat();dump(out/'state.json',state)
            if stage['exit_code']:raise RuntimeError(f'Timing stage {ordinal} failed; preserve log')
            result=read(out/f'{ordinal:02d}-{mode}/summary.json')
            print(json.dumps({'ordinal':ordinal,'mode':mode,'active_replay_seconds':result['active_replay_seconds'],'command_seconds':result['command_statistics']['command_seconds'],'all_nine_metrics_exact':True}),flush=True)
        results=[read(out/f'{i:02d}-{m}/summary.json') for i,m in enumerate(protocol['order'])]
        pairs=[{'scalar_ordinal':a,'batch_ordinal':b,'wall_ratio':results[a]['wall_seconds']/results[b]['wall_seconds'],
                'active_replay_ratio':results[a]['active_replay_seconds']/results[b]['active_replay_seconds'],
                'command_ratio':results[a]['command_statistics']['command_seconds']/results[b]['command_statistics']['command_seconds']} for a,b in protocol['pairs']]
        summary={'protocol_sha256':sha(out/'protocol.json'),'results':results,'pairs':pairs,
                 'median_paired_active_replay_ratio':statistics.median(p['active_replay_ratio'] for p in pairs),
                 'median_paired_command_ratio':statistics.median(p['command_ratio'] for p in pairs),'scope':protocol['scope'],'production_source_changed':False}
        dump(out/'summary.json',summary);state.update(status='complete',summary_sha256=sha(out/'summary.json'),completed_at_utc=datetime.now(timezone.utc).isoformat());dump(out/'state.json',state)
        print(json.dumps({'pairs':pairs,'median_active_replay_ratio':summary['median_paired_active_replay_ratio']}),flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error));dump(out/'state.json',state);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['scalar','batch']);parser.add_argument('--ordinal',type=int)
    args=parser.parse_args()
    if args.mode is None:parent()
    else:
        assert args.ordinal is not None;child(args.mode,args.ordinal)
