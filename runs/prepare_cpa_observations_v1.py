from pathlib import Path
import json,zipfile
root=Path.cwd();work=root/'runs/cpa-observation-development-v1';work.mkdir(exist_ok=False)
changes={}
def edit(name,old,new,count=1):
    if name not in changes:changes[name]=(root/name).read_text(encoding='utf-8')
    text=changes[name];assert text.count(old)==count,(name,old,text.count(old))
    changes[name]=text.replace(old,new)
edit('atc_rl/world_pool.py','progress_scale=0.0,action_reference="direct"):','progress_scale=0.0,action_reference="direct",conflict_features=False):')
edit('atc_rl/world_pool.py',"'action_reference':action_reference},","'action_reference':action_reference,'conflict_features':bool(conflict_features)},")
edit('atc_rl/world_worker.py',"\ndef worker(connection,config,directory):",'''
def observation_recipe(guidance=False, conflict_features=False):
    """Keep action/reward settings fixed while selecting optional local features."""
    from dataclasses import replace
    from atc.recipes import RECIPES
    base = RECIPES['public_route_choice_interval5'] if guidance else replace(
        RECIPES['public_weights'], decision_interval_seconds=5)
    return replace(base, conflict_features=bool(conflict_features))

def worker(connection,config,directory):''')
edit('atc_rl/world_worker.py',"""        recipe='public_route_choice_interval5' if config['guidance'] else 'onpolicy_direct_interval5'
        if not config['guidance']:
            RECIPES[recipe]=replace(RECIPES['public_weights'],decision_interval_seconds=5)
""","""        recipe='onpolicy_observation_interval5'
        RECIPES[recipe]=observation_recipe(config['guidance'], config.get('conflict_features',False))
""")
edit('atc_rl/world_worker.py','from dataclasses import asdict, replace','from dataclasses import asdict')
edit('atc_rl/world_worker.py',"'action_reference':action_reference}","'action_reference':action_reference,'conflict_features':bool(config.get('conflict_features',False))}")
edit('atc_rl/train.py',"    parser.add_argument('--filter',action='store_true')","    parser.add_argument('--filter',action='store_true')\n    parser.add_argument('--conflict-features',action='store_true',help='Append current-state closest-approach features to each local observation')")
edit('atc_rl/train.py',"            parser.error('Action reference must match the PPO comparison')","            parser.error('Action reference must match the PPO comparison')\n        if reference_config.get('conflict_features',False)!=args.conflict_features:\n            parser.error('Conflict observation features must match the PPO reference')")
edit('atc_rl/train.py',"config.update(world_seeds=seeds,actor_information='own standard aircraft observation plus remaining-time fraction',","config.update(world_seeds=seeds,actor_information=('own aircraft observation with current-state closest-approach features plus remaining-time fraction' if args.conflict_features else 'own standard aircraft observation plus remaining-time fraction'),")
edit('atc_rl/train.py','progress_scale=args.progress_scale,action_reference=args.action_reference)','progress_scale=args.progress_scale,action_reference=args.action_reference,conflict_features=args.conflict_features)')
edit('atc_rl/evaluate.py',"'action_reference':config.get('action_reference','direct'),","'action_reference':config.get('action_reference','direct'),'conflict_features':config.get('conflict_features',False),",2)
edit('atc_rl/evaluate.py',"action_reference=config.get('action_reference','direct'))","action_reference=config.get('action_reference','direct'),conflict_features=config.get('conflict_features',False))")
edit('atc_rl/curves.py',"('action_reference','direct')]","('action_reference','direct'),('conflict_features',False)]",2)
edit('atc_rl/compare.py',"    if first[1].get('action_reference','direct')!=trained[1].get('action_reference','direct'):","    if first[1].get('conflict_features',False)!=trained[1].get('conflict_features',False):\n        raise ValueError('Initial/trained comparison changed conflict_features')\n    if first[1].get('action_reference','direct')!=trained[1].get('action_reference','direct'):")
edit('atc_rl/compare.py',"    trained_manifest=json.loads((trained_parent/'checkpoints.json').read_text(encoding='utf-8-sig'))","    if trained_config.get('conflict_features',False)!=trained[1].get('conflict_features',False):\n        raise ValueError('Evaluator did not use the trained conflict_features setting')\n    trained_manifest=json.loads((trained_parent/'checkpoints.json').read_text(encoding='utf-8-sig'))")
edit('atc_rl/algorithm_compare.py',"    counts=[r['live_transitions'] for r in (ppo_record,mappo_record)]","    if ppo.get('conflict_features',False)!=mappo.get('conflict_features',False):\n        raise ValueError('Algorithm comparison changed conflict_features')\n    counts=[r['live_transitions'] for r in (ppo_record,mappo_record)]")
edit('atc_rl/algorithm_compare.py',"        if protocol.get('evaluation_reward_scale',1.)!=1.","        if protocol.get('conflict_features',False)!=first_protocol.get('conflict_features',False):\n            raise ValueError('Algorithm evaluation changed conflict_features')\n        if protocol.get('evaluation_reward_scale',1.)!=1.")
edit('atc_rl/algorithm_compare.py',"    identity=actor_match(","        if config.get('conflict_features',False)!=loaded[name][1].get('conflict_features',False):\n            raise ValueError('Evaluator ignored training conflict_features')\n    identity=actor_match(")
edit('atc_rl/matrix_evaluate.py',"    summary=read(directory/'training_summary.json');audit=read(directory/'checkpoint-audit.json')","    if config.get('conflict_features',False):raise ValueError('Existing matrix protocol does not include conflict_features')\n    summary=read(directory/'training_summary.json');audit=read(directory/'checkpoint-audit.json')")
edit('atc_rl/matrix_evaluate.py',"    expected_sources={k:v for k,v in plan['source_sha256'].items() if k.endswith('.py')}","    if protocol.get('conflict_features',False):raise ValueError('Existing matrix protocol does not include conflict_features')\n    expected_sources={k:v for k,v in plan['source_sha256'].items() if k.endswith('.py')}")
edit('jobs/train_onpolicy.slurm','if [[ "${FILTER}" == 1 ]]; then ARGS+=(--filter); fi','if [[ "${FILTER}" == 1 ]]; then ARGS+=(--filter); fi\nif [[ "${ATC_CONFLICT_FEATURES:-0}" == 1 ]]; then ARGS+=(--conflict-features); fi')
edit('tests/rl/test_algorithm_compare.py',"('entropy_coefficient',0.)])","('entropy_coefficient',0.),('conflict_features',True)])")
edit('tests/rl/test_curve_configuration.py',"['guidance','filter','action_reference','evaluation_reward_scale','evaluation_progress_scale']","['guidance','filter','conflict_features','action_reference','evaluation_reward_scale','evaluation_progress_scale']")
edit('tests/rl/test_curve_configuration.py',"if change in ('guidance','filter'):","if change in ('guidance','filter','conflict_features'):")
new_test=r'''from dataclasses import asdict
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from gymnasium import spaces
from atc.conflicts import ConflictPrediction, FEATURE_NAMES
from atc.recipes import RECIPES
from atc_rl.world_worker import observation_recipe

@pytest.mark.parametrize("guidance", [False, True])
def test_predictive_observations_change_no_reward_action_or_guidance_setting(guidance):
    original=observation_recipe(guidance,False)
    augmented=observation_recipe(guidance,True)
    before,after=asdict(original),asdict(augmented)
    assert {key for key in before if before[key]!=after[key]}=={"conflict_features"}
    reference=RECIPES["public_route_choice_interval5" if guidance else "public_weights"]
    assert original.reward_kwargs()==reference.reward_kwargs()
    assert original.decision_interval_seconds==5
    assert not reference.conflict_features

class ToyWorld:
    def __init__(self):
        self.intruder_obs=SimpleNamespace(n=3,pos_norm=10000.,spd_norm=100.)
        self.intrusion_distance=5
        self.observation_spaces={"own":spaces.Dict({name:spaces.Box(-np.inf,np.inf,(3,),dtype=np.float64)
                                                   for name in ("x_r","y_r","vx_r","vy_r")})}
    def _get_obs(self,ac_id):
        return {"x_r":np.array([3.,0.,0.]),"y_r":np.zeros(3),
                "vx_r":np.array([-3.,0.,0.]),"vy_r":np.zeros(3)}

class PredictiveToy(ConflictPrediction,ToyWorld):
    pass

def test_features_use_relative_vector_units_and_preserve_original_slots(monkeypatch):
    monkeypatch.setitem(sys.modules,"bluesky",SimpleNamespace(traf=SimpleNamespace(ntraf=2)))
    base=ToyWorld();env=PredictiveToy()
    original=base._get_obs("own");observed=env._get_obs("own")
    for name,value in original.items():np.testing.assert_array_equal(observed[name],value)
    assert observed["traffic_tcpa"][0]==pytest.approx(100/180)
    assert observed["traffic_entry_time"][0]==pytest.approx((30000-9260)/300/180)
    assert observed["traffic_dcpa"][0]==0
    assert observed["traffic_predicted_conflict"][0]==1
    for name in FEATURE_NAMES:
        assert np.all(observed[name][1:]==0)
        assert env.observation_spaces["own"][name].contains(observed[name])
    assert set(observed)==set(original)|set(FEATURE_NAMES)

def test_legacy_matrix_cannot_silently_accept_a_new_observation_experiment(tmp_path):
    import json
    from atc_rl.matrix_evaluate import validate_training
    folder=tmp_path/"runs/row";folder.mkdir(parents=True)
    (folder/"config.json").write_text(json.dumps({"conflict_features":True}))
    with pytest.raises(ValueError,match="conflict_features"):
        validate_training(tmp_path,{},{"run_dir":"runs/row"})
'''
changes['tests/rl/test_conflict_observations.py']=new_test
with zipfile.ZipFile(work/'before-source.zip','x',zipfile.ZIP_DEFLATED) as archive:
    for name in changes:
        path=root/name
        if path.exists():archive.write(path,name)
for name,text in changes.items():
    if name.endswith('.py'):compile(text,name,'exec')
for name,text in changes.items():(root/name).write_text(text,encoding='utf-8',newline='\n')
(work/'changes.json').write_text(json.dumps({'changed_files':list(changes),'scope':'Optional observation features only; existing frozen runs and cluster bundles are unchanged.'},indent=2),encoding='utf-8')
print(json.dumps({'edited_files':len(changes),'syntax_checked':True,'training_with_new_features':'Not started'}))
