"""Execute the declared command-correction feedback experiment in isolated sources."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,zipfile
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ.update(SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent;RUN=BASE/'execution'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dump=lambda p,v:p.write_text(json.dumps(v,indent=2),encoding='utf-8')
PROTOCOL=read(BASE/'protocol.json');ARCHIVE=ROOT/'output/candidates/decimal-heading-v1.zip'

def prepare(stage):
    out=RUN/stage;out.mkdir(exist_ok=False);candidate=out/'candidate'
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None;archive.extractall(candidate)
    manifest=read(candidate/'manifest.json')
    for name,digest in manifest['files'].items():assert sha(candidate/name)==digest,name
    source=candidate/'source';cache=source/'runs/simulator/cache/navdata.p'
    cache.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'runs/simulator/cache/navdata.p',cache)
    os.chdir(source);sys.path.insert(0,str(source));sys.path.insert(0,str(BASE))
    dump(out/'provenance.json',{'protocol_sha256':sha(BASE/'protocol.json'),'runner_sha256':sha(Path(__file__)),
        'extension_sha256':sha(BASE/'intervention_reward.py'),'candidate_archive_sha256':sha(ARCHIVE),
        'navigation_cache_sha256':sha(cache)})
    return out,candidate,manifest

def verify_files(candidate,manifest):
    for name,digest in manifest['files'].items():assert sha(candidate/name)==digest,name

def training(stage):
    out,candidate,manifest=prepare(stage);coefficient=float(stage[-1])
    import atc.envs as environments
    import atc.train as trainer
    from atc.heading_transport import attach_decimal_heading
    from intervention_reward import InterventionReward
    original=environments.make_env;wrappers=[];scenario_specs=[]
    def make_env(kind,*args,**kwargs):
        assert kind=='ma'
        env=attach_decimal_heading(original(kind,*args,**kwargs))
        world=env.unwrapped
        assert world.action_frequency==10 and world.distance_margin==5 and world.intrusion_distance==5
        import bluesky as bs
        from bluesky.core.entity import getproxied
        assert bs.tools.geo.kwikqdrdist.__module__=='bluesky.tools.geo._cgeo'
        assert 'openap' in type(getproxied(bs.traf.perf)).__module__.lower()
        wrapped=InterventionReward(env,coefficient);wrappers.append(wrapped)
        original_reset=wrapped.reset
        def reset(*a,**kw):
            result=original_reset(*a,**kw)
            scenario_specs.append(asdict(world.scenario))
            return result
        wrapped.reset=reset
        return wrapped
    environments.make_env=make_env
    training_dir=out/'training'
    config=PROTOCOL['training']
    sys.argv=['atc.train','--env','ma','--algorithm','sac','--recipe',config['recipe'],'--guard-traffic',
        '--workers','1','--steps','25000','--seed',str(config['seed']),'--device','cpu','--run-dir',str(training_dir),
        '--checkpoint-every','25000','--learning-starts','5000','--gradient-steps','4','--batch-size','256',
        '--buffer-size','100000','--critic-warmup-updates','0','--max-wall-seconds','0']
    dump(out/'experiment-configuration.json',{'coefficient':coefficient,'reward_formula':config['penalty'],
        'training_configuration_scope':'Read this record together with training/config.json; generic resume omits this experimental reward wrapper.',
        'protocol_sha256':sha(BASE/'protocol.json'),'extension_sha256':sha(BASE/'intervention_reward.py')})
    try:trainer.main()
    finally:
        dump(out/'reward-statistics.json',[w.statistics for w in wrappers]);dump(out/'training-scenarios.json',scenario_specs)
    checkpoint=training_dir/'checkpoints/model_25000_steps.zip'
    with zipfile.ZipFile(checkpoint) as archive:
        data=json.loads(archive.read('data'))
        assert data['num_timesteps']==25000 and data['_n_updates']==7996 and data['seed']==2910
    summary=read(training_dir/'training_summary.json');assert summary['timesteps']==25000
    assert len(wrappers)==1 and wrappers[0].statistics['commands']==summary['live_transitions']
    assert (wrappers[0].statistics['penalty']>0)==bool(coefficient)
    if coefficient:
        import io,torch
        def tensors(path):
            with zipfile.ZipFile(path) as archive:return torch.load(io.BytesIO(archive.read('policy.pth')),map_location='cpu',weights_only=True)
        a=tensors(RUN/'train_0/training/initial-model.zip');b=tensors(training_dir/'initial-model.zip')
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),'Initial policy parameters differ'
        assert scenario_specs[0]==read(RUN/'train_0/training-scenarios.json')[0]
    verify_files(candidate,manifest)
    dump(out/'completion.json',{'status':'complete','checkpoint_sha256':sha(checkpoint),'training_summary':summary,
        'coefficient':coefficient,'unchanged_packaged_files':len(manifest['files']),'initial_policy_matches_control':True if coefficient else None})

def evaluation(stage):
    out,candidate,manifest=prepare(stage);coefficient=float(stage[-1]);validation=stage.startswith('validate')
    import numpy as np
    import atc.evaluate as evaluator
    from atc import deployment
    from atc.metrics import METRICS,summarize
    from intervention_reward import InterventionReward
    model_path=candidate/'models/ma/model.zip'
    if not validation:
        model_path=out/'deployed/model.zip';model_path.parent.mkdir()
        shutil.copyfile(RUN/f'train_{int(coefficient)}/training/checkpoints/model_25000_steps.zip',model_path)
        deployment_manifest=read(candidate/'models/ma/deployment.json');deployment_manifest['model_sha256']=sha(model_path)
        dump(model_path.parent/'deployment.json',deployment_manifest)
    act=deployment.load_policy('ma',model_path)
    def predict(observations):
        actions=np.stack([act(obs) for obs in np.asarray(observations)])
        assert np.isfinite(actions).all() and (np.abs(actions)<=1).all()
        return actions
    trace=hashlib.sha256();scenarios=[];initial_states=[];wrappers=[];runtime={}
    def make_env(*args,**kwargs):
        env=deployment.make_env('ma');world=env.unwrapped
        import bluesky as bs
        from bluesky.core.entity import getproxied
        runtime.update(geo_backend=bs.tools.geo.kwikqdrdist.__module__,performance_module=type(getproxied(bs.traf.perf)).__module__)
        assert runtime['geo_backend']=='bluesky.tools.geo._cgeo' and 'openap' in runtime['performance_module'].lower()
        assert world.action_frequency==5 and world.distance_margin==5 and world.intrusion_distance==5
        if validation:env=InterventionReward(env,coefficient);wrappers.append(env)
        original_reset,original_step=env.reset,env.step
        goals=None;scenario_json=None
        def snapshot():return json.loads(json.dumps(asdict(world.scenario)))
        def reset(*a,**kw):
            nonlocal goals,scenario_json
            result=original_reset(*a,**kw);scenario=snapshot();assert len(scenario['agents'])==10
            goals={a['ac_id']:list(a['goal']) for a in scenario['agents']};scenario_json=json.dumps(scenario,sort_keys=True)
            assert {k:list(v) for k,v in world._goal.items()}==goals
            scenarios.append(scenario);initial_states.append(result[1]);return result
        def update(value):trace.update(json.dumps(value,sort_keys=True,allow_nan=False).encode())
        def step(actions):
            result=original_step(actions)
            assert {k:list(v) for k,v in world._goal.items()}==goals
            if not env.agents:assert json.dumps(snapshot(),sort_keys=True)==scenario_json
            update({k:np.asarray(v).tolist() for k,v in actions.items()})
            update({k:np.asarray(v).tolist() for k,v in result[0].items()})
            update(result[2:]);update({'ids':list(bs.traf.id),'time':float(world.sim_time)})
            for name in ('lat','lon','hdg','tas'):trace.update(np.asarray(getattr(bs.traf,name),dtype=np.float64).tobytes())
            return result
        env.reset,env.step=reset,step
        return env
    evaluator.make_env=make_env
    count=2 if validation else 20
    rows,summary=evaluator.evaluate('ma',count,2026,predict,recipe='public_route_choice_fast_residual_interval5',guard_static=True,guard_traffic=True)
    assert len(rows)==count*10 and len(scenarios)==count
    assert all(summary[k]==v for k,v in summarize(rows,count,10).items())
    with (out/'evaluation.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['episode','agent',*METRICS]);writer.writeheader()
        writer.writerows({key:r[key] for key in writer.fieldnames} for r in rows)
    summary.update(model_sha256=sha(model_path),protocol_sha256=sha(BASE/'protocol.json'),csv_sha256=sha(out/'evaluation.csv'),
        runtime=runtime,official_protocol=False,training_penalty_enabled=validation,penalty_coefficient=coefficient if validation else 0)
    dump(out/'evaluation.json',summary);dump(out/'scenarios.json',scenarios);dump(out/'initial-states.json',initial_states)
    verify_files(candidate,manifest)
    dump(out/'completion.json',{'status':'complete','all_nine_metrics_recomputed':True,'decision_boundary_trace_sha256':trace.hexdigest(),
        'unchanged_packaged_files':len(manifest['files']),'reward_statistics':[w.statistics for w in wrappers],
        'summary_sha256':sha(out/'evaluation.json')})

def parent():
    import ctypes
    class M(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total_physical','available_physical','total_pagefile','available_pagefile','total_virtual','available_virtual','available_extended_virtual')]
    m=M();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    if m.available_physical<1.5*1024**3 or shutil.disk_usage(BASE).free<1024**3:raise SystemExit('Wait for memory or disk headroom before this experiment')
    assert sha(ARCHIVE)==PROTOCOL['candidate_archive_sha256']
    assert sha(BASE/'intervention_reward.py')==PROTOCOL['extension_sha256']
    assert sha(BASE/'unit-tests.xml')==PROTOCOL['unit_test_xml_sha256']
    RUN.mkdir(exist_ok=False)
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(BASE/'protocol.json'),
          'runner_sha256':sha(Path(__file__)),'extension_sha256':sha(BASE/'intervention_reward.py'),'stage_order':PROTOCOL['stage_order']}
    dump(RUN/'execution-plan.json',plan)
    state={'status':'running','stages':[]};dump(RUN/'state.json',state)
    try:
        for stage in PROTOCOL['stage_order']:
            record={'stage':stage,'started_at_utc':datetime.now(timezone.utc).isoformat()};state['stages'].append(record)
            with (RUN/f'{stage}.log').open('x',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--stage',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                record['pid']=process.pid;dump(RUN/'state.json',state);record['exit_code']=process.wait()
            record['finished_at_utc']=datetime.now(timezone.utc).isoformat();dump(RUN/'state.json',state)
            if record['exit_code']:raise RuntimeError(f'{stage} failed; preserve logs and completed stages')
            print(json.dumps({'stage':stage,'status':'complete'}),flush=True)
            if stage=='validate_1':
                a,b=(read(RUN/f'validate_{i}/completion.json') for i in (0,1))
                assert a['decision_boundary_trace_sha256']==b['decision_boundary_trace_sha256']
                assert (RUN/'validate_0/evaluation.csv').read_bytes()==(RUN/'validate_1/evaluation.csv').read_bytes()
                assert read(RUN/'validate_0/scenarios.json')==read(RUN/'validate_1/scenarios.json')
                assert read(RUN/'validate_0/initial-states.json')==read(RUN/'validate_1/initial-states.json')
                assert a['reward_statistics'][0]['penalty']==0 and b['reward_statistics'][0]['penalty']>0
                dump(RUN/'validation-audit.json',{'status':'passed','decision_boundary_traces_exact':True,'all_nine_metric_records_exact':True,
                    'worlds':2,'aircraft':20,'feedback_nonzero':True,'penalty_sum':b['reward_statistics'][0]['penalty']})
        sys.path.insert(0,str(RUN/'evaluate_0/candidate/source'))
        from atc.compare import load_evaluation,paired_comparison
        control_meta,control=load_evaluation(RUN/'evaluate_0/evaluation');shaped_meta,shaped=load_evaluation(RUN/'evaluate_1/evaluation')
        classical=ROOT/'runs/goal-region-v1/retry2/classical_point_goal'
        assert read(RUN/'evaluate_0/scenarios.json')==read(RUN/'evaluate_1/scenarios.json')
        assert read(RUN/'evaluate_0/initial-states.json')==read(RUN/'evaluate_1/initial-states.json')
        assert read(RUN/'evaluate_0/scenarios.json')==[r['scenario'] for r in read(classical/'scenarios.json')]
        reference_meta,reference=load_evaluation(classical/'evaluation')
        assert read(RUN/'evaluate_0/initial-states.json')==read(classical/'initial-states.json')
        primary_folder=ROOT/'runs/goal-region-v1/learned-screen/learned_point_goal'
        assert read(primary_folder/'completion.json')['status']=='complete'
        assert read(RUN/'evaluate_0/scenarios.json')==[r['scenario'] for r in read(primary_folder/'scenarios.json')]
        _,primary=load_evaluation(primary_folder/'evaluation')
        def metric(meta,key):return meta['metrics'][key]['mean']
        gate_checks={}
        for label,base in [('same_seed_control',control_meta),('classical',reference_meta)]:
            gate_checks[label]={
                'arrival_no_lower':metric(shaped_meta,'waypoint_reached')>=metric(base,'waypoint_reached'),
                'clean_gain_at_least_2pp':shaped_meta['clean_completion_rate']>=base['clean_completion_rate']+.02-1e-12,
                'exposure_no_higher':all(metric(shaped_meta,k)<=metric(base,k) for k in ('intrusion_time','time_in_restricted_area','time_outside_sector')),
                'flight_increase_at_most_5_percent':metric(shaped_meta,'flight_time')<=1.05*metric(base,'flight_time')}
        result={'protocol_sha256':sha(BASE/'protocol.json'),'control_vs_feedback':paired_comparison(control,shaped,20),
            'classical_vs_feedback':paired_comparison(reference,shaped,20),'primary_vs_feedback':paired_comparison(primary,shaped,20),
            'screen_gate_checks':gate_checks,'advance_to_200_development_worlds':all(all(v.values()) for v in gate_checks.values()),
            'selected_deployment_changed':False,
            'scope':'Twenty reused development worlds, one shared training seed, all arms retained. No unseen-test or general-superiority claim.'}
        dump(RUN/'comparison.json',result)
        assert sha(BASE/'intervention_reward.py')==plan['extension_sha256'] and sha(Path(__file__))==plan['runner_sha256']
        state.update(status='complete',comparison_sha256=sha(RUN/'comparison.json'),completed_at_utc=datetime.now(timezone.utc).isoformat())
        dump(RUN/'state.json',state)
    except BaseException as error:
        state.update(status='failed',error=repr(error));dump(RUN/'state.json',state);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=PROTOCOL['stage_order']);args=parser.parse_args()
    parent() if args.stage is None else (training(args.stage) if args.stage.startswith('train') else evaluation(args.stage))
