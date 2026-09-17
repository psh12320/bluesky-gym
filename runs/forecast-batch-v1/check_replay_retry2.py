"""Validate batched initial traffic forecasts inside an isolated fixed-policy replay."""
import argparse,contextlib,faulthandler,hashlib,json,os,runpy,shutil,sys,time,zipfile
faulthandler.enable()
from datetime import datetime,timezone
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ.update(SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--env',choices=['sa','ma'],required=True);args=parser.parse_args()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
kernel=json.loads((base/'summary.json').read_text());assert kernel['all_coordinates_and_bytes_exact']
archive=root/'output/candidates/decimal-heading-v1.zip'
assert sha(archive)=='da020cf4079af110fcfcb7598b13f58e84859f8b5a0e549d9a96ad9ab5c86b2c'
out=base/f'replay-{args.env}-retry2';out.mkdir(exist_ok=False);candidate=out/'candidate';candidate.mkdir()
with zipfile.ZipFile(archive) as bundle:assert bundle.testzip() is None;bundle.extractall(candidate)
source=candidate/'source';sys.path.insert(0,str(source))
cache=root/'runs/simulator/cache/navdata.p'
assert cache.is_file()
destination=source/'runs/simulator/cache/navdata.p'
destination.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(cache,destination)
assert sha(cache)==sha(destination)
protocol={'started_at_utc':datetime.now(timezone.utc).isoformat(),'track':args.env,'seed':2026,'episodes':2,
          'candidate_archive_sha256':sha(archive),'reused_navigation_cache_sha256':sha(cache),'previous_ma_attempt':'replay-ma/runner-exit-audit.json','kernel_summary_sha256':sha(base/'summary.json'),
          'trial_script_sha256':sha(Path(__file__)),'mode':'Mandatory scalar forecast/lifetime verification at every decision',
          'scope':'Experimental batching in an isolated process. The selected deployment and all running scoring jobs remain unchanged.',
          'timing_scope':'Verification repeats scalar calculations; elapsed time is not an optimization benchmark'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
state={'status':'running','pid':os.getpid(),'protocol_sha256':sha(out/'protocol.json')}
stats={'decision_batches':0,'aircraft_forecasts_checked':0,'all_forecast_coordinates_byte_equal':True,'all_lifetimes_byte_equal':True}
def save():(out/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:stream.write(value);stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:stream.flush()
save();start=time.perf_counter()
try:
    with (out/'replay.log').open('x',encoding='utf-8') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        import numpy as np
        from atc import deployment
        original_make_env=deployment.make_env
        def make_env(*arguments,**keywords):
            print('Constructing archived environment',flush=True)
            env=original_make_env(*arguments,**keywords)
            print('Archived environment constructed',flush=True)
            # Import after the original factory initializes BlueSky's compiled backend.
            import bluesky as bs
            from atc.traffic_projection import predict_commands,capture_times
            raw=env.unwrapped
            assert hasattr(raw,'_traffic_plan_time') and hasattr(raw,'_traffic_plans')
            original_action=raw._get_action
            def action_with_batch(ac_id,action):
                decision_time=float(bs.sim.simt)
                if raw._traffic_plan_time!=decision_time:
                    agents=list(bs.traf.id);count=len(agents);assert count>0
                    positions=np.stack([raw._projection_xy((bs.traf.lat[i],bs.traf.lon[i])) for i in range(count)])
                    headings=np.asarray(bs.traf.hdg,dtype=float).copy();speeds=np.asarray(bs.traf.tas,dtype=float).copy()
                    paths=predict_commands(positions,headings,speeds,np.zeros(count),speeds)
                    plans={};stats['decision_batches']+=1
                    for i,agent in enumerate(agents):
                        expected=predict_commands(positions[i],float(headings[i]),float(speeds[i]),[0],[float(speeds[i])])
                        same=paths[i].tobytes()==expected[0].tobytes()
                        stats['all_forecast_coordinates_byte_equal'] &= same
                        assert same,('forecast changed',decision_time,agent)
                        goal=raw._goal.get(agent) if hasattr(raw,'_goal') else None
                        target=raw._projection_xy(goal) if goal is not None else None
                        life=capture_times(paths[i:i+1],target,raw.distance_margin)[0]
                        expected_life=capture_times(expected,target,raw.distance_margin)[0]
                        same_life=np.asarray(life).tobytes()==np.asarray(expected_life).tobytes()
                        stats['all_lifetimes_byte_equal'] &= same_life
                        assert same_life,('lifetime changed',decision_time,agent)
                        plans[agent]=(paths[i].copy(),life);stats['aircraft_forecasts_checked']+=1
                    raw._traffic_plans=plans;raw._traffic_plan_time=decision_time
                    if stats['decision_batches']%100==0:
                        state['prediction_checks']=stats.copy();save();print(json.dumps(stats),flush=True)
                return original_action(ac_id,action)
            raw._get_action=action_with_batch
            print('Isolated forecast hook attached',flush=True)
            return env
        deployment.make_env=make_env
        sys.argv=[str(candidate/'check_development.py'),'--env',args.env]
        runpy.run_path(str(candidate/'check_development.py'),run_name='__main__')
    check=json.loads((candidate/f'checks/{args.env}.json').read_text());assert check['all_metrics_match']
    manifest=json.loads((candidate/'manifest.json').read_text())
    for name,digest in manifest['files'].items():assert sha(candidate/name)==digest,name
    assert stats['decision_batches']>0 and stats['aircraft_forecasts_checked']>0
    result={'track':args.env,'seed':2026,'episodes':2,'wall_seconds':time.perf_counter()-start,
            'prediction_checks':stats,'development_check':check,'unchanged_packaged_files':len(manifest['files']),
            'primary_deployment_changed':False,'timing_scope':protocol['timing_scope']}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    state.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),summary_sha256=sha(out/'summary.json'))
    save();print(json.dumps(result,indent=2))
except BaseException as error:
    state.update(status='failed',error=repr(error),prediction_checks=stats);save();raise
