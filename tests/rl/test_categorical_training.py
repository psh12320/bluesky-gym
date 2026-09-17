from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from atc_rl.buffer import ActiveRolloutBuffer, RolloutAccounting
from atc_rl.exploration import configuration, log_std_initialization, ppo_options, require_matching, validate_model
from atc_rl.maneuvers import ManeuverMapping, ManeuverVecEnv, initialize_route_choice
from atc_rl.policy import AircraftPolicy
from atc_rl.pretrained import action_space_compatibility
from atc_rl.rollout_diagnostics import rollout_diagnostics
from atc_rl.world_worker import local_observation_schema
from tests.rl.test_exploration import ConstantObservation


def test_schema_is_derived_from_actual_flattening_and_appends_one_clock():
    dictionary=spaces.Dict({"airspeed":spaces.Box(0,1,(1,)),
                           "cos_drift":spaces.Box(-1,1,(1,)),
                           "intruders":spaces.Box(-1,1,(9,)),
                           "sin_drift":spaces.Box(-1,1,(1,))})
    schema=local_observation_schema(dictionary)
    sample={key:np.full(space.shape,index,dtype=space.dtype) for index,(key,space) in enumerate(dictionary.spaces.items())}
    flattened=spaces.flatten(dictionary,sample)
    for field in schema[:-1]:
        np.testing.assert_array_equal(flattened[field["offset"]:field["offset"]+field["size"]],sample[field["name"]])
    assert schema[-1]=={"name":"time_remaining_fraction","offset":12,"size":1}
    mapping=ManeuverMapping.from_schema(schema)
    assert mapping==ManeuverMapping(13,1,11)
    assert ManeuverMapping.from_specification(mapping.specification())==mapping
    with pytest.raises(ValueError):local_observation_schema(spaces.Box(-1,1,(12,)))
    with pytest.raises(ValueError):local_observation_schema(spaces.Dict({"time_remaining_fraction":spaces.Box(0,1,(1,))}))


@pytest.mark.parametrize("change",[
    {"version":2},{"speed_choices":[0,1,2]},{"default_deterministic_choice":[10,2]},
    {"heading_choices":list(range(20))},{"mapping_changes_rewards":True}])
def test_altered_maneuver_specifications_are_rejected(change):
    mapping=ManeuverMapping(3,0,1)
    with pytest.raises(ValueError):ManeuverMapping.from_specification({**mapping.specification(),**change})


@pytest.mark.parametrize("extra",[
    {"sde_weight_std":.05},{"sde_sample_freq":12},{"neutral_action_mean":True},{"action_reference":"goal_offset"}])
def test_categorical_configuration_rejects_double_mapping_or_gaussian_initialization(extra):
    with pytest.raises(ValueError):configuration({"exploration":"categorical",**extra})


def test_categorical_comparisons_require_the_same_mapping():
    mapping=ManeuverMapping(3,0,1)
    config={"exploration":"categorical","maneuver_mapping":mapping.specification()}
    require_matching(config,config)
    require_matching(config,{"exploration":"categorical"},allow_unbuilt=True)
    with pytest.raises(ValueError):require_matching(config,{"exploration":"categorical"})
    with pytest.raises(ValueError):require_matching(config,{**config,"maneuver_mapping":ManeuverMapping(3,1,0).specification()})
    assert ppo_options(config)=={"use_sde":False,"sde_sample_freq":-1}
    assert log_std_initialization(config)==0


class MaskedCommands(ConstantObservation):
    action_space=spaces.Box(-1,1,(2,),dtype=np.float64)
    def __init__(self,padded=False):
        super().__init__()
        self.padded=padded
        self.counter=0
    def reset(self,seed=None,options=None):
        self.counter=0
        return super().reset(seed=seed,options=options)
    def step(self,action):
        self.counter+=1
        inactive=self.padded and self.counter%2==0
        observation={"actor":np.full(3,.2,np.float32),"critic":np.full(8,.4,np.float32)}
        reward=0. if inactive else float(1-np.square(action).sum())
        return observation,reward,inactive,False,{"inactive":inactive,"aircraft_done":False}


@pytest.mark.parametrize("centralized",[False,True])
def test_masked_categorical_ppo_learns_reloads_and_preserves_live_accounting(tmp_path,centralized):
    torch.set_num_threads(1)
    mapping=ManeuverMapping(3,0,1)
    env=ManeuverVecEnv(DummyVecEnv([lambda:MaskedCommands(False),lambda:MaskedCommands(True)]),mapping)
    config={"exploration":"categorical","maneuver_mapping":mapping.specification()}
    model=PPO(AircraftPolicy,env,seed=84,device="cpu",n_steps=4,batch_size=4,n_epochs=1,
              rollout_buffer_class=ActiveRolloutBuffer,**ppo_options(config),
              policy_kwargs={"centralized":centralized,"actor_width":8,"critic_width":8})
    initialize_route_choice(model)
    before={k:v.clone() for k,v in model.policy.state_dict().items()}
    accounting=RolloutAccounting()
    model.learn(16,callback=accounting)
    assert model.num_timesteps==16 and accounting.live_transitions==12 and accounting.padded_transitions==4
    assert any(not torch.equal(v,before[k]) for k,v in model.policy.state_dict().items() if k.startswith("action_net"))
    diagnostics=rollout_diagnostics(model.rollout_buffer)
    assert diagnostics["rollout_live_samples"]==6
    assert diagnostics["rollout_action_box_clip_fraction_live"] is None
    path=tmp_path/"model.zip"
    model.save(path)
    restored=PPO.load(path,device=model.device)
    validate_model(restored,config)
    probe={"actor":np.full((2,3),.2,np.float32),"critic":np.zeros((2,8),np.float32)}
    first=restored.predict(probe,deterministic=True)[0]
    probe["critic"].fill(9)
    np.testing.assert_array_equal(first,restored.predict(probe,deterministic=True)[0])
    with pytest.raises(ValueError):validate_model(restored,{"initial_action_std":.05})
    with pytest.raises(ValueError):validate_model(restored,{**config,"maneuver_mapping":ManeuverMapping(4,0,1).specification()})
    indices=model.rollout_buffer.valid_indices
    model.rollout_buffer.actions[indices[0],0]=20
    with pytest.raises(ValueError,match="indices"):rollout_diagnostics(model.rollout_buffer)
    env.close()


def test_pretraining_allows_only_identical_registered_categorical_actions():
    expected=spaces.MultiDiscrete([20,3])
    result=action_space_compatibility(expected,expected)
    assert result["kind"]=="categorical" and result["nvec"]==[20,3]
    for other in (spaces.MultiDiscrete([19,3]),spaces.MultiDiscrete([20,3],start=[1,0]),spaces.Box(-1,1,(2,))):
        with pytest.raises(ValueError):action_space_compatibility(expected,other)
