"""Inspect the exact frozen runtime bindings and fixed competition parameters."""
from pathlib import Path
import os,sys,json,hashlib,inspect,subprocess
from datetime import datetime,timezone
for k,v in {'SDL_VIDEODRIVER':'dummy','PYGAME_HIDE_SUPPORT_PROMPT':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}.items():os.environ.setdefault(k,v)
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from atc import submission
from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_zoo.competition_v0 import CompetitionZooEnv
from core.scenario import ScenarioGenerator
from core.actions import HeadingAction,SpeedAction
import bluesky as bs
import numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fixed_sources=['core/scenario.py','core/actions.py','bluesky_gym/envs/competition_env.py','bluesky_zoo/competition/competition.py']
source_hashes={}
for name in fixed_sources:
    original=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','show','HEAD:'+name])
    actual=(ROOT/name).read_bytes()
    assert actual.replace(b'\r\n',b'\n')==original.replace(b'\r\n',b'\n'),name
    source_hashes[name]=sha(ROOT/name)
results={}
for track,base,freeze_path,interval in [('sa',CompetitionEnv,'runs/heldout-2027-sa-reach250-v1/protocol.json',10),('ma',CompetitionZooEnv,'runs/heldout-2027-ma-interval5-v1/protocol.json',5)]:
    frozen=json.loads((ROOT/freeze_path).read_text())
    model=ROOT/frozen['model']
    assert sha(model)==frozen['model_sha256']
    act=submission.load_policy(track,model)
    env=submission.make_env(track)
    world=env.unwrapped
    try:
        bindings={}
        for method in ('step','_update_metrics','_get_info','_init_metrics'):
            bound=getattr(world,method).__func__
            assert bound is getattr(base,method),method
            bindings[method]=bound.__module__+'.'+bound.__qualname__
        parameters={k:getattr(world,k) for k in ('episode_time_limit','intrusion_distance','distance_margin','ac_spd','altitude','center','n_obstacles','sim_dt','action_frequency')}
        defaults=inspect.signature(base.__init__).parameters
        for k in ('episode_time_limit','intrusion_distance','distance_margin','ac_spd','altitude','center','n_obstacles'):
            assert parameters[k]==defaults[k].default,k
        assert world.sim_dt==1 and world.action_frequency==interval and world._fixed_scenario is None
        assert type(world.scenario_generator) is ScenarioGenerator
        expected=ScenarioGenerator(n_agents=1 if track=='sa' else 10,n_intruders=10 if track=='sa' else 0)
        assert vars(world.scenario_generator)==vars(expected)
        assert type(world.heading_action) is HeadingAction and type(world.speed_action) is SpeedAction
        observation,_=env.reset(seed=2026)
        assert len(world.scenario.agents)==(1 if track=='sa' else 10)
        assert len(world.scenario.intruder_routes)==(10 if track=='sa' else 0)
        assert world.intruder_obs.n==9
        rng_before=repr(world._np_random.bit_generator.state)
        if track=='sa':
            action=act(observation)
            assert action.shape==(2,)
            _,_,_,_,infos=env.step(action)
            first_times=[infos['flight_time']]
        else:
            actions={a:act(observation[a]) for a in env.agents}
            assert all(a.shape==(2,) for a in actions.values())
            _,_,_,_,infos=env.step(actions)
            first_times=[r['flight_time'] for r in infos.values()]
        assert first_times==[float(interval)]*(1 if track=='sa' else 10)
        assert rng_before==repr(world._np_random.bit_generator.state)
        assert all(str(t)=='A320' for t in bs.traf.type)
        assert 'RESO OFF' in inspect.getsource(base.reset)
        results[track]={'model_sha256':sha(model),'base_scoring_and_step_bindings':bindings,'fixed_parameters':parameters,'generator_configuration':vars(world.scenario_generator),'scenario_agents':len(world.scenario.agents),'scripted_intruders':len(world.scenario.intruder_routes),'observed_traffic_slots':world.intruder_obs.n,'action_dimensions':2,'action_classes':['HeadingAction','SpeedAction'],'first_decision_flight_times':first_times,'first_decision_scenario_rng_unchanged':True,'aircraft_types':sorted(set(map(str,bs.traf.type))),'conflict_resolution_runtime_class':type(bs.traf.cr).__module__+'.'+type(bs.traf.cr).__qualname__,'original_reset_contains_reso_off':True}
    finally:env.close()
output={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':source_hashes,'fixed_source_matches_original_head':True,'checks':results,'scope':'Exact frozen model/configuration runtime bindings, all fixed constructor/generator parameters and one development decision per track. Complements full metric replays; not a claim of operational safety.','coordination':'One shared neural policy; the joint MA command filter uses centralized live traffic state and processes aircraft in a stable order. Not fully decentralized execution.'}
(Path(__file__).resolve().parent/'audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
