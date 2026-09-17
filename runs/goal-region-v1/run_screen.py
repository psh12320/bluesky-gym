"""Compare the predeclared MA classical point-goal and goal-region controllers."""
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
PROTOCOL=read(BASE/'protocol.json')

def child(mode):
    out=BASE/mode;out.mkdir(exist_ok=False);candidate=out/'candidate';candidate.mkdir()
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
    if mode=='classical_goal_region':
        from goal_region import install
        install()
    original_make_env=evaluation.make_env;scenarios=[];initial_states=[];runtime={}
    def make_env(*args,**kwargs):
        env=original_make_env(*args,**kwargs)
        import bluesky as bs
        from atc.heading_transport import attach_decimal_heading
        attach_decimal_heading(env)
        backend=bs.tools.geo.kwikqdrdist.__module__
        assert backend=='bluesky.tools.geo._cgeo',backend
        world=env.unwrapped
        assert world.distance_margin==5 and world.action_frequency==5 and world.intrusion_distance==5
        runtime.update(geo_backend=backend,performance_module=type(bs.traf.perf).__module__,
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
    rows,summary=evaluation.evaluate('ma',20,2026,None,recipe='public_route_choice_interval5',
                                    guard_static=True,guard_traffic=True,track_goal=True,goal_speed_action=1.0)
    assert len(rows)==200 and len(scenarios)==20
    expected=summarize(rows,20,10)
    assert all(summary[key]==value for key,value in expected.items())
    with (out/'evaluation.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['episode','agent',*METRICS]);writer.writeheader()
        writer.writerows({key:r[key] for key in writer.fieldnames} for r in rows)
    summary.update(mode=mode,policy='goal',recipe='public_route_choice_interval5',guard_static=True,guard_traffic=True,
        official_protocol=False,model_sha256=None,goal_speed_action=1.0,heading_transport='plain_decimal',
        experimental_goal_region=mode=='classical_goal_region',protocol_sha256=sha(BASE/'protocol.json'),
        execution_plan_sha256=sha(BASE/'execution-plan.json'),candidate_archive_sha256=sha(ARCHIVE),
        extension_sha256=sha(BASE/'goal_region.py'),csv_sha256=sha(out/'evaluation.csv'),runtime=runtime,
        all_scenarios_and_goals_unchanged=True)
    dump(out/'evaluation.json',summary);dump(out/'scenarios.json',scenarios);dump(out/'initial-states.json',initial_states)
    for name,digest in manifest['files'].items():assert sha(candidate/name)==digest,name
    dump(out/'completion.json',{'status':'complete','summary_sha256':sha(out/'evaluation.json'),
                              'scenarios_sha256':sha(out/'scenarios.json'),'unchanged_packaged_files':len(manifest['files'])})

def parent():
    assert sha(ARCHIVE)==PROTOCOL['candidate_archive_sha256']
    suite=ET.parse(BASE/'geometry-tests.xml').getroot()
    suites=list(suite.iter('testsuite'))
    assert sum(int(s.attrib.get('tests',0)) for s in suites)==9
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    plan=BASE/'execution-plan.json';assert not plan.exists()
    dump(plan,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(BASE/'protocol.json'),
        'runner_sha256':sha(Path(__file__)),'extension_sha256':sha(BASE/'goal_region.py'),
        'geometry_test_source_sha256':sha(BASE/'test_goal_region.py'),'geometry_tests_xml_sha256':sha(BASE/'geometry-tests.xml'),
        'geometry_tests_passed':9,'arms':PROTOCOL['initial_arms'],'arm_order_fixed_before_results':True})
    state={'status':'running','execution_plan_sha256':sha(plan),'stages':[]};dump(BASE/'state.json',state)
    try:
        for mode in PROTOCOL['initial_arms']:
            stage={'mode':mode,'started_at_utc':datetime.now(timezone.utc).isoformat()};state['stages'].append(stage)
            with (BASE/f'{mode}.log').open('x',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--mode',mode],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                stage['pid']=process.pid;dump(BASE/'state.json',state);stage['exit_code']=process.wait()
            stage['finished_at_utc']=datetime.now(timezone.utc).isoformat();dump(BASE/'state.json',state)
            if stage['exit_code']:raise RuntimeError(f'{mode} failed; retain the log')
            summary=read(BASE/mode/'evaluation.json')
            print(json.dumps({'mode':mode,'arrival':summary['metrics']['waypoint_reached']['mean'],'flight':summary['metrics']['flight_time']['mean'],'clean':summary['clean_completion_rate']}),flush=True)
        reference,changed=PROTOCOL['initial_arms']
        a,b=(read(BASE/m/'scenarios.json') for m in (reference,changed));assert a==b,'Scenario sequence changed'
        assert read(BASE/reference/'initial-states.json')==read(BASE/changed/'initial-states.json')
        sys.path.insert(0,str(BASE/'reference/source'))
        from atc.compare import load_evaluation,paired_comparison
        from atc.metrics import METRICS
        original_meta,original=load_evaluation(BASE/reference/'evaluation')
        changed_meta,modified=load_evaluation(BASE/changed/'evaluation')
        historical_meta,historical=load_evaluation(ROOT/'runs/goal-route-choice-interval5-v1-fast-joint-ma-20')
        old={(r['episode'],r['agent']):r for r in historical}
        historical_changes={k:sum(r[k]!=old[(r['episode'],r['agent'])][k] for r in original) for k in METRICS}
        comparison={'protocol_sha256':sha(BASE/'protocol.json'),'execution_plan_sha256':sha(plan),
            'all_20_scenario_specs_and_initial_metrics_identical':True,'summary_by_arm':{reference:original_meta,changed:changed_meta},
            'metrics':paired_comparison(original,modified,20),'historical_point_goal_changed_records':historical_changes,
            'scope':'Paired development screen with 20 worlds and classical controllers. No training, learned advantage or unseen-test claim.',
            'selected_deployment_changed':False}
        dump(BASE/'comparison.json',comparison)
        assert sha(BASE/'goal_region.py')==read(plan)['extension_sha256']
        state.update(status='complete',comparison_sha256=sha(BASE/'comparison.json'),completed_at_utc=datetime.now(timezone.utc).isoformat());dump(BASE/'state.json',state)
        print(json.dumps({'status':'complete','comparison_sha256':state['comparison_sha256'],'historical_changed_records':historical_changes}),flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error));dump(BASE/'state.json',state);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=PROTOCOL['initial_arms']);args=parser.parse_args()
    parent() if args.mode is None else child(args.mode)
