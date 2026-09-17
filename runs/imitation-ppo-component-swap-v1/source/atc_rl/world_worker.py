"""One process per BlueSky world with isolated simulator files."""
from pathlib import Path
import os
import traceback
import numpy as np

POPULATION=10

def pack_observations(observations,live,agents,observation_dim,time_remaining):
    """Local actor input plus critic-only joint observations, alive flags and identity."""
    if not 0<=time_remaining<=1:raise ValueError('Remaining-time fraction outside [0, 1]')
    active=set(live)
    local=np.zeros((len(agents),observation_dim+1),dtype=np.float32)
    mask=np.array([agent in active for agent in agents],dtype=np.float32)
    for index,agent in enumerate(agents):
        if agent in active:
            local[index,:observation_dim]=np.asarray(observations[agent],dtype=np.float32)
            local[index,-1]=time_remaining
    joint=np.concatenate((local.reshape(-1),mask))
    context=np.concatenate((np.tile(joint,(len(agents),1)),np.eye(len(agents),dtype=np.float32)),axis=1)
    return {'actor':local,'critic':context}

def observation_recipe(guidance=False, conflict_features=False):
    """Keep action/reward settings fixed while selecting optional local features."""
    from dataclasses import replace
    from atc.recipes import RECIPES
    base = RECIPES['public_route_choice_interval5'] if guidance else replace(
        RECIPES['public_weights'], decision_interval_seconds=5)
    return replace(base, conflict_features=bool(conflict_features))

def worker(connection,config,directory):
    env=None
    try:
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
        os.environ.update(SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);os.chdir(directory)
        import hashlib
        import json
        import shutil
        from dataclasses import asdict
        import bluesky as bs
        runtime_directory=directory/'simulator'
        runtime_directory.mkdir(exist_ok=True)
        # Copy an existing local navigation cache; workers never write a shared cache.
        cache=Path(__file__).resolve().parents[1]/'runs/simulator/cache/navdata.p'
        cache_hash=None
        if cache.is_file():
            target=runtime_directory/'cache/navdata.p'
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(cache,target)
            cache_hash=hashlib.sha256(target.read_bytes()).hexdigest()
        bs.init(mode='sim',detached=True,workdir=str(runtime_directory))
        from atc.recipes import RECIPES
        from atc.envs import make_env
        from atc.heading_transport import attach_decimal_heading
        recipe='onpolicy_observation_interval5'
        RECIPES[recipe]=observation_recipe(config['guidance'], config.get('conflict_features',False))
        from atc_rl.action_support import guard_settings, verify_guard_selection
        guard_static,guard_traffic=guard_settings(config)
        env=attach_decimal_heading(make_env('ma',recipe,guard_static,guard_traffic))
        effective_guards=verify_guard_selection(env,config)
        if config.get('mask_conflict_features',False):
            from atc_rl.feature_mask import MaskConflictFeatures
            env=MaskConflictFeatures(env)
        from atc_rl.traffic_scaling import position_scale, ScaleTrafficMA
        traffic_scale=position_scale(config)
        if traffic_scale!=1.0:env=ScaleTrafficMA(env,traffic_scale)
        action_reference=config.get('action_reference','direct')
        if action_reference=='goal_offset':
            from atc_rl.goal_offset import GoalOffsetActions
            env=GoalOffsetActions(env)
        elif action_reference!='direct':raise ValueError('Unknown action reference')
        progress_scale=config.get('progress_scale',0.0)
        if progress_scale:
            from atc_rl.progress_reward import PotentialProgress
            env=PotentialProgress(env,progress_scale)
        world=env.unwrapped
        agents=list(env.possible_agents)
        assert len(agents)==POPULATION and world.action_frequency==5
        assert world.episode_time_limit==3000 and world.distance_margin==5 and world.intrusion_distance==5
        import bluesky as bs
        from bluesky.core.entity import getproxied
        backend=bs.tools.geo.kwikqdrdist.__module__
        assert backend=='bluesky.tools.geo._cgeo',backend
        performance=type(getproxied(bs.traf.perf)).__module__
        assert 'openap' in performance.lower(),performance
        observation_space=env.observation_space(agents[0]);action_space=env.action_space(agents[0])
        dim=observation_space.shape[0]
        steps={agent:0 for agent in agents}
        learning_returns={agent:0.0 for agent in agents}
        runtime={**effective_guards,'geo_backend':backend,'performance_module':performance,'population':len(agents),
                 'decision_interval_seconds':5,'episode_time_limit':3000,'actor_addition':'time_remaining_fraction',
                 'worker_imported_torch': 'torch' in __import__('sys').modules,
                 'isolated_simulator_files':True,'training_progress_scale':progress_scale,
                 'action_reference':action_reference,'conflict_features':bool(config.get('conflict_features',False)),
                 'mask_conflict_features':bool(config.get('mask_conflict_features',False)),
                 'traffic_position_scale':traffic_scale}
        (directory/'runtime.json').write_text(json.dumps({**runtime,'simulator_directory':str(runtime_directory),
            'copied_navdata_sha256':cache_hash},indent=2),encoding='utf-8')
        connection.send(('ready',(observation_space,action_space,runtime)))
        def reset(seed=None):
            obs,infos=env.reset(seed=seed)
            scenario_data=json.dumps(asdict(world.scenario),sort_keys=True,separators=(',',':'),default=float)
            scenario_hash=hashlib.sha256(scenario_data.encode()).hexdigest()
            for agent in agents:
                steps[agent]=0
                learning_returns[agent]=0.0
                infos[agent]['initial_potential']=env.potentials[agent] if progress_scale else 0.0
                infos[agent]['scenario_sha256']=scenario_hash
            return pack_observations(obs,agents,agents,dim,1.0),[infos[a] for a in agents]
        while True:
            command,payload=connection.recv()
            if command=='reset':connection.send(('ok',reset(payload)))
            elif command=='step':
                active=list(env.agents)
                actions={agent:payload[i] for i,agent in enumerate(agents) if agent in active}
                obs,reward,term,trunc,info=env.step(actions)
                remaining=max(0.0,1-world.sim_time/world.episode_time_limit)
                survivors=[a for a in active if not term[a]]
                packed=pack_observations(obs,survivors,agents,dim,remaining)
                rewards=np.zeros(POPULATION,dtype=np.float32)
                dones=np.ones(POPULATION,dtype=bool)
                infos=[]
                world_done=not env.agents
                for i,agent in enumerate(agents):
                    valid=agent in active
                    item={'inactive':not valid,'aircraft_done':False,'TimeLimit.truncated':False,'world_completed':bool(world_done and i==0)}
                    if valid:
                        steps[agent]+=1;rewards[i]=reward[agent]
                        learning_returns[agent]+=float(reward[agent])
                        item['native_reward']=env.last_native_rewards[agent] if progress_scale else float(reward[agent])
                        item['shaping_reward']=env.last_shaping[agent] if progress_scale else 0.0
                        dones[i]=term[agent] or trunc[agent]
                        item.update(aircraft_done=bool(dones[i]),aircraft_terminated=bool(term[agent]),
                                    aircraft_truncated=bool(trunc[agent]))
                        if dones[i]:
                            metrics={k:float(v) for k,v in info[agent].items()}
                            item['metrics']=metrics
                            item['learning_episode_return']=learning_returns[agent]
                            item['episode']={'r':learning_returns[agent],'l':steps[agent]}
                            terminal_actor=np.concatenate((np.asarray(obs[agent],dtype=np.float32),[remaining])).astype(np.float32)
                            item['terminal_observation']={'actor':terminal_actor,'critic':packed['critic'][i].copy()}
                            # The competition deadline is a task terminal, not an
                            # artificial training cutoff. Do not bootstrap past 3000 s.
                    infos.append(item)
                if not env.agents:
                    packed,reset_infos=reset()
                else:reset_infos=[{} for _ in agents]
                connection.send(('ok',(packed,rewards,dones,infos,reset_infos)))
            elif command=='goal_actions':
                from atc.baselines import goal_tracker
                predict=goal_tracker(world.observation_space(agents[0]),speed_action=1.0)
                connection.send(('ok',predict(payload)))
            elif command=='getattr':connection.send(('ok',getattr(world,payload)))
            elif command=='close':break
            else:raise ValueError(f'Unsupported worker command {command}')
    except BaseException:
        try:connection.send(('error',traceback.format_exc()))
        except (BrokenPipeError,EOFError,OSError):pass
    finally:
        if env is not None:env.close()
        connection.close()
