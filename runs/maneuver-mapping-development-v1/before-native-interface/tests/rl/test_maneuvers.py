import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from atc_rl.maneuvers import ManeuverMapping,ManeuverVecEnv,ManeuverActor,initialize_route_choice
from atc_rl.policy import AircraftPolicy
from tests.rl.test_exploration import ConstantObservation


def mapping():return ManeuverMapping(3,0,1)


def test_all_heading_and_speed_choices_are_exact_bounded_commands():
    m=mapping();x=np.array([1.,0.,1.],dtype=np.float32)
    choices=np.array([[h,s] for h in range(20) for s in range(3)],dtype=np.int64)
    obs=np.tile(x,(60,1));commands=m.decode(obs,choices)
    assert commands.shape==(60,2) and commands.dtype==np.float32
    assert np.isfinite(commands).all() and np.max(np.abs(commands))<=1
    np.testing.assert_array_equal(m.decode(obs,m.encode_teacher(obs,commands)),commands)
    np.testing.assert_array_equal(m.decode(x,np.array([0,2])),[0.,1.])
    np.testing.assert_array_equal(m.decode(x,np.array([1,0])),[-1.,-1.])
    np.testing.assert_array_equal(m.decode(x,np.array([19,1])),[1.,0.])


@pytest.mark.parametrize("drift",[-180.,-90.,-25.,0.,17.,90.,180.])
def test_dynamic_option_uses_observed_bearing_and_same_float32_mapping(drift):
    m=mapping();x=np.array([np.cos(np.deg2rad(drift)),np.sin(np.deg2rad(drift)),.5],dtype=np.float32)
    expected=np.float32(np.clip(-np.arctan2(x[1],x[0])/(np.pi/4),-1,1))
    result=m.decode(x,np.array([0,2],dtype=np.int64))
    assert result[0]==expected and result[1]==1
    np.testing.assert_array_equal(result,m.decode(x[None],np.array([[0,2]]))[0])


@pytest.mark.parametrize("choices",[[-1,0],[20,0],[0,3],[0.,2.],[0,2,1],[True,False]])
def test_invalid_indices_cannot_reach_simulator(choices):
    with pytest.raises(ValueError):mapping().decode([1,0,.5],choices)


def test_schema_requires_contiguous_unique_scalar_drift_fields():
    schema=[dict(name="cos_drift",offset=0,size=1),dict(name="sin_drift",offset=1,size=1),dict(name="time_remaining_fraction",offset=2,size=1)]
    assert ManeuverMapping.from_schema(schema)==mapping()
    for bad in (schema[:1],schema+[schema[-1]],schema[1:]):
        with pytest.raises(ValueError):ManeuverMapping.from_schema(bad)
    for bad in ([1,np.nan,.5],[[1,0]],[[[1,0,.5]]]):
        with pytest.raises(ValueError):mapping().heading_options(bad)


class RecordedWorld(ConstantObservation):
    action_space=spaces.Box(-1,1,(2,),dtype=np.float64)
    def step(self,action):
        self.actions.append(action.copy())
        observation=self.reset()[0]
        return observation,7.,False,False,{"original_info":True}


def test_vector_mapping_preserves_rewards_observations_and_infos():
    world=RecordedWorld();base=DummyVecEnv([lambda:world]);env=ManeuverVecEnv(base,mapping())
    with pytest.raises(RuntimeError):env.step_async(np.array([[0,2]]))
    observation=env.reset();expected=mapping().decode(observation["actor"],np.array([[0,2]]))
    returned,reward,done,info=env.step(np.array([[0,2]]))
    np.testing.assert_array_equal(world.actions[-1],expected[0])
    assert reward.tolist()==[7.] and done.tolist()==[False] and info[0]["original_info"]
    assert returned.keys()==observation.keys()
    env.close()


@pytest.mark.parametrize("centralized",[False,True])
def test_categorical_ppo_training_reload_and_decentralized_native_action(tmp_path,centralized):
    torch.set_num_threads(1)
    env=ManeuverVecEnv(DummyVecEnv([RecordedWorld]),mapping())
    model=PPO(AircraftPolicy,env,seed=733,device="cpu",n_steps=4,batch_size=4,n_epochs=1,
              learning_rate=3e-5,policy_kwargs={"centralized":centralized,"actor_width":8,"critic_width":8})
    initialize_route_choice(model)
    observations=env.reset()
    choices=model.predict(observations,deterministic=True)[0]
    np.testing.assert_array_equal(choices,[[0,2]])
    distributions=model.policy.get_distribution({k:torch.as_tensor(v) for k,v in observations.items()}).distribution
    assert distributions[0].probs[0,0].item()==pytest.approx(.9)
    assert distributions[1].probs[0,2].item()==pytest.approx(.9)
    before={k:v.clone() for k,v in model.policy.state_dict().items()}
    model.learn(8)
    assert model.num_timesteps==8 and model._n_updates==2
    assert any(not torch.equal(v,before[k]) for k,v in model.policy.state_dict().items())
    path=tmp_path/"policy.zip";model.save(path)
    restored=PPO.load(path,device=model.device)
    actor=ManeuverActor(restored,mapping())
    local=observations["actor"][0]
    action=actor(local)
    expected=mapping().decode(local,restored.predict({"actor":local,"critic":np.zeros(8,dtype=np.float32)},deterministic=True)[0])
    np.testing.assert_array_equal(action,expected)
    context=torch.zeros((1,8),requires_grad=True)
    probe={"actor":torch.as_tensor(local[None]),"critic":context}
    logits=restored.policy.get_distribution(probe).distribution[0].logits
    gradient=torch.autograd.grad(logits.sum(),context)[0]
    assert torch.count_nonzero(gradient)==0
    value=restored.policy.predict_values(probe)
    critic_gradient=torch.autograd.grad(value.sum(),context)[0]
    assert bool(torch.count_nonzero(critic_gradient))==centralized
    with pytest.raises(ValueError):initialize_route_choice(restored)
    env.close()
