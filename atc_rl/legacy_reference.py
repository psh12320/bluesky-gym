"""Evaluate the preserved SAC deployment without changing its source or weights."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify_candidate(candidate):
    manifest=read(candidate/'manifest.json')
    for name,digest in manifest['files'].items():
        path=(candidate/name).resolve()
        if not path.is_relative_to(candidate) or not path.is_file() or sha(path)!=digest:
            raise ValueError('Candidate integrity check failed: '+name)
    model=candidate/'models/ma/model.zip'
    deployment=read(model.parent/'deployment.json')
    if deployment['environment_configuration']['algorithm']!='sac':
        raise ValueError('This comparison requires the preserved SAC deployment')
    if sha(model)!=deployment['model_sha256']:
        raise ValueError('SAC deployment model hash mismatch')
    return model,deployment,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['learned','zero-residual'],default='learned')
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--seed',type=int,default=20260)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.episodes<1 or args.seed<0 or args.seed in (42,20301,20302):
        parser.error('Use a positive episode count and a development seed')
    root=Path(__file__).resolve().parents[1]
    candidate=args.candidate.resolve();directory=args.out.resolve()
    if not directory.is_relative_to(root/'runs') or directory.exists():
        parser.error('Choose a fresh output directory under runs/')
    model_path,deployment,manifest=verify_candidate(candidate)
    if args.preflight_only:
        print(json.dumps({'candidate_files_verified':len(manifest['files']),
            'model_sha256':sha(model_path),'simulation_started':False}));return
    source=candidate/'source'
    for name in ('atc','core','bluesky_gym','bluesky_zoo'):
        if name in sys.modules:raise RuntimeError('Candidate must load in a fresh process')
    sys.path.insert(0,str(source))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    os.environ.update(SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
    directory.mkdir(parents=True);os.chdir(directory)
    import numpy as np
    import bluesky as bs
    runtime=directory/'simulator';runtime.mkdir()
    cache=root/'runs/simulator/cache/navdata.p'
    if not cache.is_file():raise FileNotFoundError('A verified local navigation cache is required')
    (runtime/'cache').mkdir();shutil.copyfile(cache,runtime/'cache/navdata.p')
    bs.init(mode='sim',detached=True,workdir=str(runtime))
    from atc import deployment as adapter
    from atc.metrics import METRICS,summarize
    from bluesky.core.entity import getproxied
    import atc,core,bluesky_gym,bluesky_zoo
    for module in (atc,core,bluesky_gym,bluesky_zoo):
        if source not in Path(module.__file__).resolve().parents:
            raise RuntimeError('Candidate source isolation failed: '+module.__name__)
    predict=adapter.load_policy('ma',model_path)
    environment=adapter.make_env('ma',10)
    sources={name.removeprefix('source/'):digest for name,digest in manifest['files'].items()
             if name.startswith('source/') and name.endswith('.py')}
    protocol={'algorithm':'sac','mode':args.mode,'seed':args.seed,'episodes':args.episodes,
        'guidance':True,'filter':True,'action_reference':'legacy_fast_residual',
        'candidate':str(candidate),'model_path':str(model_path),'model_sha256':sha(model_path),
        'deployment':deployment,'source_sha256':sources,'runner_sha256':sha(Path(__file__)),
        'evaluation_reward_scale':1.0,'evaluation_progress_scale':0.0,
        'additional_training_transitions':0,'original_counted_training_budget':deployment['environment_configuration']['steps'],
        'inference':'deterministic per-aircraft predictions using the frozen deployment loader',
        'scenario_protocol':'seed once, then continue the generator stream',
        'zero_residual_interpretation':'Controller reference, not an evaluated untrained neural checkpoint',
        'comparison_limit':'Historical SAC has different action mapping, supports, architecture and training budget; system reference only, not an equal-budget algorithm ranking.'}
    (directory/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=[];scenarios=[];decisions=0;started=time.perf_counter()
    try:
        world=environment.unwrapped
        assert world.action_frequency==5 and world.episode_time_limit==3000
        assert world.distance_margin==5 and world.intrusion_distance==5
        assert len(environment.possible_agents)==10
        assert bs.tools.geo.kwikqdrdist.__module__=='bluesky.tools.geo._cgeo'
        performance=type(getproxied(bs.traf.perf)).__module__
        assert 'openap' in performance.lower()
        with (directory/'aircraft.csv').open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['episode','scenario_sha256','agent',*METRICS]);writer.writeheader()
            for episode in range(args.episodes):
                observations,_=environment.reset(seed=args.seed if episode==0 else None)
                serialized=json.dumps(asdict(world.scenario),sort_keys=True,separators=(',',':'),default=float)
                scenario=hashlib.sha256(serialized.encode()).hexdigest()
                scenarios.append({'episode':episode,'sha256':scenario})
                while environment.agents:
                    actions={agent:(predict(observations[agent]) if args.mode=='learned' else np.zeros(2,dtype=np.float32))
                             for agent in environment.agents}
                    observations,_,terminated,truncated,infos=environment.step(actions);decisions+=1
                    for agent,info in infos.items():
                        if terminated[agent] or truncated[agent]:
                            row={'episode':episode,'scenario_sha256':scenario,'agent':agent,
                                 **{key:float(info[key]) for key in METRICS}}
                            writer.writerow(row);records.append(row);stream.flush();os.fsync(stream.fileno())
                print(json.dumps({'episode':episode,'completed_aircraft':len(records)}),flush=True)
        summary=summarize(records,args.episodes,10)
        verify_candidate(candidate)
        assert sha(Path(__file__))==protocol['runner_sha256']
        summary.update(algorithm='sac',mode=args.mode,seed=args.seed,world_decisions=decisions,
            wall_seconds=time.perf_counter()-started,csv_sha256=sha(directory/'aircraft.csv'),
            model_sha256=sha(model_path),execution_sources_unchanged=True,additional_training_transitions=0,
            runtime={'geo_backend':bs.tools.geo.kwikqdrdist.__module__,'performance_module':performance,
                     'navdata_sha256':sha(cache),'isolated_simulator_files':True})
        (directory/'scenarios.json').write_text(json.dumps(scenarios,indent=2),encoding='utf-8')
        (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary),flush=True)
    finally:
        environment.close()


if __name__=='__main__':main()