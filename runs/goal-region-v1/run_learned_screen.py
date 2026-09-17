"""Compare the same frozen MA policy under point-goal and goal-region routing."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,zipfile
import xml.etree.ElementTree as ET
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ.update(SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dump=lambda p,v:p.write_text(json.dumps(v,indent=2),encoding='utf-8')
ARCHIVE=ROOT/'output/candidates/decimal-heading-v1.zip'
PROTOCOL=read(BASE/'learned-protocol.json')
RUN=BASE/'learned-screen'

def child(mode):
    out=RUN/mode;out.mkdir(exist_ok=False);candidate=out/'candidate';candidate.mkdir()
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None;archive.extractall(candidate)
    manifest=read(candidate/'manifest.json')
    for name,digest in manifest['files'].items():assert sha(candidate/name)==digest,name
    source=candidate/'source';cache=source/'runs/simulator/cache/navdata.p';cache.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'runs/simulator/cache/navdata.p',cache)
    os.chdir(source);sys.path.insert(0,str(source));sys.path.insert(0,str(BASE))
    import numpy as np
    import atc.evaluate as evaluation
    from atc.metrics import METRICS,summarize
    if mode=='learned_goal_region':
        from goal_region import install
        install()
    from atc import deployment
    model_path=candidate/'models/ma/model.zip'
    act=deployment.load_policy('ma',model_path)
    assert sha(model_path)==PROTOCOL['model_sha256']
    def predict(observations):
        values=np.asarray(observations)
        result=act(values) if values.ndim==1 else np.stack([act(obs) for obs in values])
        assert np.isfinite(result).all() and (np.abs(result)<=1).all()
        return result
    original_make_env=lambda *args,**kwargs:deployment.make_env('ma')
    scenarios=[];initial_states=[];runtime={}
    def make_env(*args,**kwargs):
        env=original_make_env(*args,**kwargs)
        import bluesky as bs
        from bluesky.core.entity import getproxied
        backend=bs.tools.geo.kwikqdrdist.__module__
        assert backend=='bluesky.tools.geo._cgeo',backend
        world=env.unwrapped
        assert world.distance_margin==5 and world.action_frequency==5 and world.intrusion_distance==5
        runtime.update(geo_backend=backend,performance_module=type(getproxied(bs.traf.perf)).__module__,
                       distance_margin_km=world.distance_margin,action_interval_seconds=world.action_frequency,
                       intrusion_distance_nm=world.intrusion_distance)
        assert 'openap' in runtime['performance_module'].lower()
        original_reset,original_step=env.reset,env.step
        expected_goals={};scenario_hash=None
        def snapshot():return json.loads(json.dumps(asdict(world.scenario)))
        def assert_original_goals():
            actual={key:list(value) for key,value in world._goal.items()}
            assert actual==expected_goals,'Assigned waypoint changed'
        def reset(*args,**kwargs):
            nonlocal expected_goals,scenario_hash
            observation,infos=original_reset(*args,**kwargs)
            scenario=snapshot();assert len(scenario['agents'])==10
            expected_goals={a['ac_id']:list(a['goal']) for a in scenario['agents']}
            assert_original_goals()
            scenario_hash=hashlib.sha256(json.dumps(scenario,sort_keys=True).encode()).hexdigest()
            scenarios.append({'episode':len(scenarios),'sha256':scenario_hash,'scenario':scenario})
            initial_states.append({a:{k:float(info[k]) for k in METRICS} for a,info in infos.items()})
            return observation,infos
        def step(actions):
            result=original_step(actions)
            assert_original_goals()
            if not env.agents:
                assert hashlib.sha256(json.dumps(snapshot(),sort_keys=True).encode()).hexdigest()==scenario_hash
            return result
        env.reset,env.step=reset,step
        return env
    evaluation.make_env=make_env
    rows,summary=evaluation.evaluate('ma',20,2026,predict,recipe='public_route_choice_fast_residual_interval5',
                                    guard_static=True,guard_traffic=True,track_goal=False)
    assert len(rows)==200 and len(scenarios)==20
    expected=summarize(rows,20,10)
    assert all(summary[key]==value for key,value in expected.items())
    with (out/'evaluation.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['episode','agent',*METRICS]);writer.writeheader()
        writer.writerows({key:r[key] for key in writer.fieldnames} for r in rows)
    summary.update(mode=mode,policy='sac',recipe='public_route_choice_fast_residual_interval5',guard_static=True,guard_traffic=True,
        official_protocol=False,model_sha256=sha(model_path),goal_speed_action=None,heading_transport='plain_decimal',
        experimental_goal_region=mode=='learned_goal_region',protocol_sha256=sha(BASE/'learned-protocol.json'),
        execution_plan_sha256=sha(RUN/'execution-plan.json'),candidate_archive_sha256=sha(ARCHIVE),
        extension_sha256=sha(BASE/'goal_region.py'),csv_sha256=sha(out/'evaluation.csv'),runtime=runtime,
        all_scenarios_and_goals_unchanged=True)
    dump(out/'evaluation.json',summary);dump(out/'scenarios.json',scenarios);dump(out/'initial-states.json',initial_states)
    for name,digest in manifest['files'].items():assert sha(candidate/name)==digest,name
    dump(out/'completion.json',{'status':'complete','summary_sha256':sha(out/'evaluation.json'),
                              'scenarios_sha256':sha(out/'scenarios.json'),'unchanged_packaged_files':len(manifest['files'])})

def parent():
    assert read(BASE/'retry2/state.json')['status']=='complete','Finish the classical screen first'
    assert sha(BASE/'goal_region.py')==PROTOCOL['extension_sha256']
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(name,ctypes.c_uint64) for name in ('total_physical','available_physical','total_pagefile','available_pagefile','total_virtual','available_virtual','available_extended_virtual')]
    memory=MemoryStatus();memory.length=ctypes.sizeof(memory)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    if memory.available_physical<1.5*1024**3:raise SystemExit('Wait for a running job to finish: less than 1.5 GiB physical memory available')
    RUN.mkdir(exist_ok=False)
    assert sha(ARCHIVE)==PROTOCOL['candidate_archive_sha256']
    suite=ET.parse(BASE/'geometry-tests.xml').getroot()
    suites=list(suite.iter('testsuite'))
    assert sum(int(s.attrib.get('tests',0)) for s in suites)==9
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    plan=RUN/'execution-plan.json';assert not plan.exists()
    dump(plan,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(BASE/'learned-protocol.json'),
        'runner_sha256':sha(Path(__file__)),'extension_sha256':sha(BASE/'goal_region.py'),
        'geometry_test_source_sha256':sha(BASE/'test_goal_region.py'),'geometry_tests_xml_sha256':sha(BASE/'geometry-tests.xml'),
        'classical_comparison_sha256':sha(BASE/'retry2/comparison.json'),'geometry_tests_passed':9,'arms':PROTOCOL['initial_arms'],'arm_order_fixed_before_results':True})
    state={'status':'running','execution_plan_sha256':sha(plan),'stages':[]};dump(RUN/'state.json',state)
    try:
        for mode in PROTOCOL['initial_arms']:
            stage={'mode':mode,'started_at_utc':datetime.now(timezone.utc).isoformat()};state['stages'].append(stage)
            with (RUN/f'{mode}.log').open('x',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--mode',mode],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                stage['pid']=process.pid;dump(RUN/'state.json',state);stage['exit_code']=process.wait()
            stage['finished_at_utc']=datetime.now(timezone.utc).isoformat();dump(RUN/'state.json',state)
            if stage['exit_code']:raise RuntimeError(f'{mode} failed; retain the log')
            summary=read(RUN/mode/'evaluation.json')
            print(json.dumps({'mode':mode,'arrival':summary['metrics']['waypoint_reached']['mean'],'flight':summary['metrics']['flight_time']['mean'],'clean':summary['clean_completion_rate']}),flush=True)
        reference,changed=PROTOCOL['initial_arms']
        a,b=(read(RUN/m/'scenarios.json') for m in (reference,changed));assert a==b,'Scenario sequence changed'
        assert read(RUN/reference/'initial-states.json')==read(RUN/changed/'initial-states.json')
        sys.path.insert(0,str(BASE/'reference/source'))
        from atc.compare import load_evaluation,paired_comparison
        from atc.metrics import METRICS
        original_meta,original=load_evaluation(RUN/reference/'evaluation')
        changed_meta,modified=load_evaluation(RUN/changed/'evaluation')
        assert a==read(BASE/'retry2/classical_point_goal/scenarios.json')
        assert read(RUN/reference/'initial-states.json')==read(BASE/'retry2/classical_point_goal/initial-states.json')
        _,classical_point=load_evaluation(BASE/'retry2/classical_point_goal/evaluation')
        _,classical_region=load_evaluation(BASE/'retry2/classical_goal_region/evaluation')
        comparison={'protocol_sha256':sha(BASE/'learned-protocol.json'),'execution_plan_sha256':sha(plan),
            'all_20_scenario_specs_and_initial_metrics_identical':True,'summary_by_arm':{reference:original_meta,changed:changed_meta},
            'metrics':paired_comparison(original,modified,20),'classical_vs_learned_point_goal':paired_comparison(classical_point,original,20),
            'classical_vs_learned_goal_region':paired_comparison(classical_region,modified,20),
            'classical_comparison_sha256':sha(BASE/'retry2/comparison.json'),
            'scope':'Paired transfer screen with 20 development worlds and the same frozen learned model. Classical comparisons included for both routing choices. No new training or unseen-test claim.',
            'selected_deployment_changed':False}
        dump(RUN/'comparison.json',comparison)
        assert sha(BASE/'goal_region.py')==read(plan)['extension_sha256']
        state.update(status='complete',comparison_sha256=sha(RUN/'comparison.json'),completed_at_utc=datetime.now(timezone.utc).isoformat());dump(RUN/'state.json',state)
        print(json.dumps({'status':'complete','comparison_sha256':state['comparison_sha256'],'model_sha256':PROTOCOL['model_sha256']}),flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error));dump(RUN/'state.json',state);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=PROTOCOL['initial_arms']);args=parser.parse_args()
    parent() if args.mode is None else child(args.mode)
