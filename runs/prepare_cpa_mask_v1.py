from pathlib import Path
import json,zipfile
root=Path.cwd();work=root/'runs/cpa-mask-development-v1';work.mkdir(exist_ok=False)
changes={}
def edit(name,old,new,count=1):
    if name not in changes:changes[name]=(root/name).read_text(encoding='utf-8')
    assert changes[name].count(old)==count,(name,old,changes[name].count(old))
    changes[name]=changes[name].replace(old,new)
changes['atc_rl/feature_mask.py']=r'''"""A same-size zero-input control for the conflict-observation experiment."""
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper
from atc.conflicts import FEATURE_NAMES

class MaskConflictFeatures(BaseParallelWrapper):
    """Zero only added conflict features, retaining spaces, actions and scoring."""
    def __init__(self, environment):
        super().__init__(environment)
        self.indices = {}
        for agent in environment.possible_agents:
            dictionary = environment.unwrapped.observation_space(agent)
            if not isinstance(dictionary, spaces.Dict) or not set(FEATURE_NAMES).issubset(dictionary.spaces):
                raise ValueError("Conflict-feature masking requires the augmented observation")
            offsets = []
            cursor = 0
            for name, space in dictionary.spaces.items():
                size = spaces.flatdim(space)
                if name in FEATURE_NAMES:
                    offsets.extend(range(cursor, cursor + size))
                cursor += size
            if environment.observation_space(agent).shape != (cursor,):
                raise ValueError("Apply conflict-feature masking after observation flattening")
            self.indices[agent] = np.asarray(offsets, dtype=np.int64)

    def _mask(self, observations):
        masked = {}
        for agent, value in observations.items():
            result = np.asarray(value).copy()
            result[self.indices[agent]] = 0
            masked[agent] = result
        return masked

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._mask(observations), infos

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        return self._mask(observations), rewards, terminated, truncated, infos
'''
edit('atc_rl/world_pool.py','conflict_features=False):','conflict_features=False,mask_conflict_features=False):')
edit('atc_rl/world_pool.py','        self.action_reference=action_reference','        if mask_conflict_features and not conflict_features:raise ValueError("Masking requires conflict_features")\n        self.action_reference=action_reference')
edit('atc_rl/world_pool.py',"'conflict_features':bool(conflict_features)},","'conflict_features':bool(conflict_features),'mask_conflict_features':bool(mask_conflict_features)},")
edit('atc_rl/world_worker.py',"        action_reference=config.get('action_reference','direct')","        if config.get('mask_conflict_features',False):\n            from atc_rl.feature_mask import MaskConflictFeatures\n            env=MaskConflictFeatures(env)\n        action_reference=config.get('action_reference','direct')")
edit('atc_rl/world_worker.py',"'conflict_features':bool(config.get('conflict_features',False))}","'conflict_features':bool(config.get('conflict_features',False)),\n                 'mask_conflict_features':bool(config.get('mask_conflict_features',False))}")
edit('atc_rl/train.py',"    parser.add_argument('--progress-scale',type=float,default=0.0)","    parser.add_argument('--mask-conflict-features',action='store_true',help='Same-size zero-input control; requires --conflict-features')\n    parser.add_argument('--progress-scale',type=float,default=0.0)")
edit('atc_rl/train.py',"    args=parser.parse_args()","    args=parser.parse_args()\n    if args.mask_conflict_features and not args.conflict_features:parser.error('Masking requires --conflict-features')")
edit('atc_rl/train.py',"            parser.error('Conflict observation features must match the PPO reference')","            parser.error('Conflict observation features must match the PPO reference')\n        if reference_config.get('mask_conflict_features',False)!=args.mask_conflict_features:\n            parser.error('Conflict feature masking must match the PPO reference')")
edit('atc_rl/train.py',"actor_information=('own aircraft observation with current-state closest-approach features plus remaining-time fraction' if args.conflict_features else 'own standard aircraft observation plus remaining-time fraction'),","actor_information=('own aircraft observation with zeroed conflict-feature channels plus remaining-time fraction' if args.mask_conflict_features else 'own aircraft observation with current-state closest-approach features plus remaining-time fraction' if args.conflict_features else 'own standard aircraft observation plus remaining-time fraction'),")
edit('atc_rl/train.py','conflict_features=args.conflict_features)','conflict_features=args.conflict_features,mask_conflict_features=args.mask_conflict_features)')
edit('atc_rl/evaluate.py',"'conflict_features':config.get('conflict_features',False),","'conflict_features':config.get('conflict_features',False),'mask_conflict_features':config.get('mask_conflict_features',False),")
edit('atc_rl/evaluate.py',"conflict_features=config.get('conflict_features',False))","conflict_features=config.get('conflict_features',False),mask_conflict_features=config.get('mask_conflict_features',False))")
edit('atc_rl/evaluate.py',"conflict_features=config.get('conflict_features',False),seed=args.seed","conflict_features=config.get('conflict_features',False),mask_conflict_features=config.get('mask_conflict_features',False),seed=args.seed")
edit('atc_rl/curves.py',"('conflict_features',False)]","('conflict_features',False),('mask_conflict_features',False)]",2)
edit('atc_rl/compare.py',"    if first[1].get('conflict_features',False)!=trained[1].get('conflict_features',False):","    if first[1].get('mask_conflict_features',False)!=trained[1].get('mask_conflict_features',False):\n        raise ValueError('Initial/trained comparison changed mask_conflict_features')\n    if first[1].get('conflict_features',False)!=trained[1].get('conflict_features',False):")
edit('atc_rl/compare.py',"    if trained_config.get('conflict_features',False)!=trained[1].get('conflict_features',False):","    if trained_config.get('mask_conflict_features',False)!=trained[1].get('mask_conflict_features',False):\n        raise ValueError('Evaluator did not use the trained mask_conflict_features setting')\n    if trained_config.get('conflict_features',False)!=trained[1].get('conflict_features',False):")
edit('atc_rl/algorithm_compare.py',"    if ppo.get('conflict_features',False)!=mappo.get('conflict_features',False):","    if ppo.get('mask_conflict_features',False)!=mappo.get('mask_conflict_features',False):\n        raise ValueError('Algorithm comparison changed mask_conflict_features')\n    if ppo.get('conflict_features',False)!=mappo.get('conflict_features',False):")
edit('atc_rl/algorithm_compare.py',"        if protocol.get('conflict_features',False)!=first_protocol.get('conflict_features',False):","        if protocol.get('mask_conflict_features',False)!=first_protocol.get('mask_conflict_features',False):\n            raise ValueError('Algorithm evaluation changed mask_conflict_features')\n        if protocol.get('conflict_features',False)!=first_protocol.get('conflict_features',False):")
edit('atc_rl/algorithm_compare.py',"        if config.get('conflict_features',False)!=loaded[name][1].get('conflict_features',False):","        if config.get('mask_conflict_features',False)!=loaded[name][1].get('mask_conflict_features',False):\n            raise ValueError('Evaluator ignored training mask_conflict_features')\n        if config.get('conflict_features',False)!=loaded[name][1].get('conflict_features',False):")
edit('atc_rl/matrix_evaluate.py',"if config.get('conflict_features',False):","if config.get('conflict_features',False) or config.get('mask_conflict_features',False):")
edit('atc_rl/matrix_evaluate.py',"if protocol.get('conflict_features',False):","if protocol.get('conflict_features',False) or protocol.get('mask_conflict_features',False):")
edit('jobs/train_onpolicy.slurm','if [[ "${ATC_CONFLICT_FEATURES:-0}" == 1 ]]; then ARGS+=(--conflict-features); fi','if [[ "${ATC_CONFLICT_FEATURES:-0}" == 1 ]]; then ARGS+=(--conflict-features); fi\nif [[ "${ATC_MASK_CONFLICT_FEATURES:-0}" == 1 ]]; then ARGS+=(--mask-conflict-features); fi')
edit('tests/rl/test_algorithm_compare.py',"('conflict_features',True)])","('conflict_features',True),('mask_conflict_features',True)])")
edit('tests/rl/test_curve_configuration.py',"'conflict_features','action_reference'","'conflict_features','mask_conflict_features','action_reference'")
edit('tests/rl/test_curve_configuration.py',"('guidance','filter','conflict_features')","('guidance','filter','conflict_features','mask_conflict_features')")
changes['tests/rl/test_feature_mask.py']=r'''import numpy as np
import pytest
from gymnasium import spaces
from pettingzoo import ParallelEnv
from atc.conflicts import FEATURE_NAMES
from atc.envs import FlattenObservations
from atc_rl.feature_mask import MaskConflictFeatures
from atc_rl.world_pool import WorldPool

class ExampleWorld(ParallelEnv):
    possible_agents=["a"]
    def __init__(self):
        self.agents=["a"]
        self.dictionary=spaces.Dict({"cos_drift":spaces.Box(-1,1,(1,),dtype=np.float64),
                                     "x_r":spaces.Box(-100,100,(9,),dtype=np.float64),
                                     **{name:spaces.Box(0,4,(9,),dtype=np.float64) for name in FEATURE_NAMES}})
    def observation_space(self,agent):return self.dictionary
    def action_space(self,agent):return spaces.Box(-1,1,(2,),dtype=np.float32)
    def reset(self,seed=None,options=None):
        self.observation={name:np.full(space.shape,.5) for name,space in self.dictionary.spaces.items()}
        self.info={"a":{"marker":17}}
        return {"a":self.observation},self.info
    def step(self,actions):
        self.received=actions
        obs,_=self.reset()
        self.scoring=({"a":-3.},{"a":False},{"a":False},{"a":{"metric":7.}})
        return obs,*self.scoring

def test_mask_retains_spaces_and_only_zeros_the_added_channels():
    original=ExampleWorld();flat=FlattenObservations(original);masked=MaskConflictFeatures(flat)
    assert masked.observation_space("a")==flat.observation_space("a")
    observation,info=masked.reset()
    restored=spaces.unflatten(original.dictionary,observation["a"])
    for name in FEATURE_NAMES:assert np.count_nonzero(restored[name])==0
    np.testing.assert_array_equal(restored["x_r"],original.observation["x_r"])
    np.testing.assert_array_equal(restored["cos_drift"],original.observation["cos_drift"])
    assert info is original.info and np.all(original.observation["traffic_dcpa"]==.5)
    assert masked.observation_space("a").contains(observation["a"])

def test_mask_does_not_modify_actions_rewards_terminals_or_scoring():
    original=ExampleWorld();masked=MaskConflictFeatures(FlattenObservations(original))
    masked.reset();actions={"a":np.array([.2,-.1],dtype=np.float32)}
    result=masked.step(actions)
    assert original.received is actions
    assert all(result[i+1] is item for i,item in enumerate(original.scoring))

def test_mask_rejects_wrong_wrapper_order_and_missing_feature_flag(tmp_path):
    with pytest.raises(ValueError,match="flattening"):MaskConflictFeatures(ExampleWorld())
    with pytest.raises(ValueError,match="requires conflict_features"):
        WorldPool(1,tmp_path,mask_conflict_features=True)
'''
with zipfile.ZipFile(work/'before-source.zip','x',zipfile.ZIP_DEFLATED) as z:
    for name in changes:
        if (root/name).exists():z.write(root/name,name)
for name,text in changes.items():
    if name.endswith('.py'):compile(text,name,'exec')
for name,text in changes.items():(root/name).write_text(text,encoding='utf-8',newline='\n')
(work/'changes.json').write_text(json.dumps({'files':list(changes),'scope':'Same-size zero-input control; no classical controller or frozen source changed.'},indent=2),encoding='utf-8')
print(json.dumps({'edited_files':len(changes),'syntax_checked':True}))
