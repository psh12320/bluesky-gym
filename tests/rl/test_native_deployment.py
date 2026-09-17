from types import SimpleNamespace
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
from atc_rl.deployment import DecentralizedActor, LocalInputsSA, GoalOffsetSA, actor_space, timed_observation
from atc_rl.goal_offset import heading_commands
from atc_rl.competition import validate_protocol


class ToySingle(gym.Env):
    def __init__(self):
        from atc.conflicts import FEATURE_NAMES
        self.observation_space = spaces.Dict({
            "cos_drift":spaces.Box(-1.,1.,(1,),dtype=np.float64),
            "sin_drift":spaces.Box(-1.,1.,(1,),dtype=np.float64),
            "speed":spaces.Box(-np.inf,np.inf,(1,),dtype=np.float64),
            **{key:spaces.Box(0.,4.,(9,),dtype=np.float64) for key in FEATURE_NAMES}})
        self.action_space = spaces.Box(-1.,1.,(2,),dtype=np.float64)
        self.agent="KL001";self.metrics={self.agent:{"flight_time":0.}}
        self.episode_time_limit=3000;self.d_heading=45
        self.last_action=None
        self.info={"native_metric":17}
    def observe(self):
        from atc.conflicts import FEATURE_NAMES
        return {"cos_drift":np.array([.6]),"sin_drift":np.array([.8]),"speed":np.array([2.]),
                **{key:np.ones(9) for key in FEATURE_NAMES}}
    def reset(self,seed=None,options=None):
        self.metrics[self.agent]["flight_time"]=0.
        return self.observe(),self.info
    def step(self,action):
        self.last_action=np.asarray(action).copy()
        self.metrics[self.agent]["flight_time"]=3000.
        return self.observe(),7.,False,True,self.info


def test_single_transfer_masks_only_added_channels_and_preserves_native_step():
    world=ToySingle()
    env=LocalInputsSA(gym.wrappers.FlattenObservation(world),mask_conflict_features=True)
    obs,info=env.reset()
    assert info is world.info and obs[-1]==1. and obs.dtype==np.float32
    assert np.count_nonzero(obs[:-1])==3
    result=env.step(np.array([.1,.2]))
    assert result[0][-1]==0 and result[1:]==(7.,False,True,world.info)
    np.testing.assert_array_equal(world.last_action,[.1,.2])
    assert env.observation_space.contains(result[0])


def test_single_heading_mapping_matches_multiagent_representation_exactly():
    world=ToySingle()
    base=LocalInputsSA(gym.wrappers.FlattenObservation(world))
    env=GoalOffsetSA(base)
    with pytest.raises(RuntimeError,match="Reset"):env.step(np.zeros(2))
    observation,_=env.reset()
    action=np.array([.15,-.3],dtype=np.float32)
    expected=heading_commands([action],[observation],env.cosine_index,env.sine_index)[0]
    output=env.step(action)
    np.testing.assert_array_equal(world.last_action,expected)
    assert world.last_action[1]==action[1]
    assert output[1:]==(7.,False,True,world.info)


def test_clock_matches_collector_float32_encoding_and_rejects_negative_time():
    original=spaces.Box(-np.inf,np.inf,(3,),dtype=np.float64)
    world=SimpleNamespace(sim_time=1235,episode_time_limit=3000)
    actual=timed_observation(np.array([.123456789,3.,-7.]),world)
    expected=np.concatenate((np.array([.123456789,3.,-7.],dtype=np.float32),
                             [1-1235/3000])).astype(np.float32)
    np.testing.assert_array_equal(actual,expected)
    assert actor_space(original).contains(actual)
    world.sim_time=-1
    with pytest.raises(ValueError,match="time"):timed_observation(np.zeros(3),world)


def test_actor_interface_supplies_no_joint_state_and_uses_scalar_inference():
    class Model:
        observation_space=spaces.Dict({"actor":spaces.Box(-np.inf,np.inf,(3,),dtype=np.float32),
                                       "critic":spaces.Box(-np.inf,np.inf,(40,),dtype=np.float32)})
        action_space=spaces.Box(-1.,1.,(2,),dtype=np.float32)
        def predict(self,inputs,deterministic):
            assert deterministic
            assert inputs["actor"].shape==(3,)
            assert inputs["critic"].shape==(40,) and not inputs["critic"].any()
            return np.array([.25,-.5],dtype=np.float32),None
    actor=DecentralizedActor(Model(),{}, {})
    np.testing.assert_array_equal(actor([1.,2.,3.]),[.25,-.5])
    with pytest.raises(ValueError):actor(np.zeros((2,3)))
    with pytest.raises(ValueError):actor([1,np.nan,3])


@pytest.mark.parametrize("seed,episodes,official,heldout",[
    (42,20,False,False),(42,20,True,False),(20260,1000,True,False),
    (20301,20,False,False),(20302,20,False,False),(-1,20,False,False),(20260,0,False,False)])
def test_wrong_official_or_unseen_protocol_cannot_run_accidentally(seed,episodes,official,heldout):
    with pytest.raises(ValueError):validate_protocol(seed,episodes,official,heldout)


@pytest.mark.parametrize("seed,episodes,official,heldout",[
    (20260,2,False,False),(42,1000,True,False),(20301,200,False,True)])
def test_explicit_registered_protocols_are_accepted(seed,episodes,official,heldout):
    validate_protocol(seed,episodes,official,heldout)

@pytest.mark.parametrize("elapsed", [float("nan"), float("inf"), -1.])
def test_nonfinite_or_negative_single_agent_clock_is_rejected(elapsed):
    world = ToySingle()
    env = LocalInputsSA(gym.wrappers.FlattenObservation(world))
    env.reset()
    world.metrics[world.agent]["flight_time"] = elapsed
    with pytest.raises(ValueError, match="time"):
        env.observation(gym.spaces.flatten(world.observation_space, world.observe()))


def test_single_agent_clock_uses_flight_time_and_resets_each_episode():
    world = ToySingle()
    assert not hasattr(world, "sim_time")
    env = LocalInputsSA(gym.wrappers.FlattenObservation(world))
    env.reset()
    world.metrics[world.agent]["flight_time"] = 1235.
    result = env.observation(gym.spaces.flatten(world.observation_space, world.observe()))
    assert result[-1] == np.float32(1 - 1235 / 3000)
    env.step(np.zeros(2))
    assert env.reset()[0][-1] == 1.


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_native_actor_matches_float32_world_transport_before_action_projection(algorithm):
    from atc_rl.deployment import LocalSACActor
    from atc_rl.world_pool import WorldPool
    original = np.array([.1234567890123, -.314159265359], dtype=np.float64)
    class Model:
        action_space = spaces.Box(-1., 1., (2,), dtype=np.float64)
        local_space = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        observation_space = (local_space if algorithm == "sac" else spaces.Dict({
            "actor": local_space, "critic": spaces.Box(-np.inf, np.inf, (40,), dtype=np.float32)}))
        def predict(self, inputs, deterministic):
            assert deterministic
            return original, None
    received = []
    pool = object.__new__(WorldPool)
    pool.waiting = False
    pool.num_envs = 10
    pool.connections = [SimpleNamespace(send=received.append)]
    pool.step_async(np.tile(original, (10, 1)))
    actor_type = DecentralizedActor if algorithm == "ppo" else LocalSACActor
    deployed = actor_type(Model(), {}, {})(np.zeros(3))
    assert deployed.dtype == np.float32
    assert received[0][0] == "step"
    np.testing.assert_array_equal(deployed, received[0][1][0])
    np.testing.assert_array_equal(original, [.1234567890123, -.314159265359])
    # A projection writes its corrected turn into the received action's dtype.
    for action in (deployed, received[0][1][0]):
        action[0] = 17.5 / 45.
    np.testing.assert_array_equal(deployed, received[0][1][0])
