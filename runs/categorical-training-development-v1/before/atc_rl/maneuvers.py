"""Local maneuver choices for categorical PPO, mapped to native heading/speed commands."""
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper

HEADING_CHOICES=20
SPEED_CHOICES=3


@dataclass(frozen=True)
class ManeuverMapping:
    actor_dim: int
    cosine_index: int
    sine_index: int

    def __post_init__(self):
        values=(self.actor_dim,self.cosine_index,self.sine_index)
        if any(type(value) is not int for value in values) or self.actor_dim<2:
            raise ValueError("Expected integer local observation dimensions and indices")
        if not 0<=self.cosine_index<self.actor_dim or not 0<=self.sine_index<self.actor_dim or self.cosine_index==self.sine_index:
            raise ValueError("Heading-reference indices must be distinct and inside the local observation")

    @classmethod
    def from_schema(cls,schema):
        fields={};cursor=0
        for field in schema:
            if field["name"] in fields or type(field["size"]) is not int or field["size"]<1 or field["offset"]!=cursor:
                raise ValueError("Observation schema must have unique, contiguous fields")
            fields[field["name"]]=field
            cursor+=field["size"]
        if any(name not in fields or fields[name]["size"]!=1 for name in ("cos_drift","sin_drift")):
            raise ValueError("Expected scalar cosine and sine of observed heading drift")
        return cls(cursor,fields["cos_drift"]["offset"],fields["sin_drift"]["offset"])

    @property
    def action_space(self):
        return spaces.MultiDiscrete([HEADING_CHOICES,SPEED_CHOICES])

    def heading_options(self,observations):
        local=np.asarray(observations,dtype=np.float32)
        if local.ndim not in (1,2) or local.shape[-1]!=self.actor_dim or not np.isfinite(local).all():
            raise ValueError("Expected finite local aircraft observations with the registered layout")
        single=local.ndim==1
        if single:local=local[None,:]
        nominal=np.clip(-np.arctan2(local[:,self.sine_index],local[:,self.cosine_index])/(np.pi/4),-1,1)
        grid=np.broadcast_to(np.linspace(-1,1,19,dtype=np.float32),(len(local),19))
        options=np.column_stack((nominal,grid)).astype(np.float32)
        return options[0] if single else options

    def decode(self,observations,choices):
        options=self.heading_options(observations)
        actions=np.asarray(choices)
        single=options.ndim==1
        expected=(2,) if single else (len(options),2)
        if actions.shape!=expected or not np.issubdtype(actions.dtype,np.integer):
            raise ValueError("Maneuver choices must be integer heading/speed indices")
        if np.any(actions<0) or np.any(actions[...,0]>=HEADING_CHOICES) or np.any(actions[...,1]>=SPEED_CHOICES):
            raise ValueError("Maneuver choice is out of range")
        if single:options,actions=options[None,:],actions[None,:]
        command=np.column_stack((options[np.arange(len(options)),actions[:,0]],
                                 np.array([-1.,0.,1.],dtype=np.float32)[actions[:,1]])).astype(np.float32)
        return command[0] if single else command

    def encode_teacher(self,observations,commands):
        """Nearest recorded label for supervised fitting; never used at deployment."""
        options=self.heading_options(observations);target=np.asarray(commands,dtype=np.float32)
        single=options.ndim==1
        expected=(2,) if single else (len(options),2)
        if target.shape!=expected or not np.isfinite(target).all() or np.any(np.abs(target)>1):
            raise ValueError("Expected finite normalized heading/speed labels")
        if single:options,target=options[None,:],target[None,:]
        heading=np.argmin(np.abs(options-target[:,0,None]),axis=1)
        speed=np.argmin(np.abs(np.array([-1.,0.,1.],dtype=np.float32)[None,:]-target[:,1,None]),axis=1)
        choices=np.column_stack((heading,speed)).astype(np.int64)
        return choices[0] if single else choices

    def specification(self):
        return {"version":1,"actor_dim":self.actor_dim,"cosine_index":self.cosine_index,"sine_index":self.sine_index,
                "heading_choices":["observed_route_bearing"]+np.linspace(-45,45,19).tolist(),
                "speed_choices":[-1,0,1],"default_deterministic_choice":[0,2],
                "navigation_prior":"Category zero follows the bearing in the existing local observation; no planner or other-aircraft state is queried.",
                "command_dtype":"float32","mapping_changes_rewards":False}


class ManeuverVecEnv(VecEnvWrapper):
    """Change only the action representation; preserve WorldPool observations and terminals."""
    def __init__(self,environment,mapping):
        if getattr(environment,"action_reference","direct")!="direct":
            raise ValueError("Maneuvers require native direct commands, not another heading mapping")
        action=environment.action_space
        if not isinstance(action,spaces.Box) or action.shape!=(2,) or not np.all(action.low==-1) or not np.all(action.high==1):
            raise ValueError("Wrapped simulator must accept normalized heading/speed commands")
        if environment.observation_space["actor"].shape!=(mapping.actor_dim,):
            raise ValueError("Maneuver observation layout differs from the simulator")
        super().__init__(environment,action_space=mapping.action_space)
        self.mapping=mapping;self._local=None

    def _cache(self,observations):
        self._local=np.asarray(observations["actor"],dtype=np.float32).copy()
        if self._local.shape!=(self.num_envs,self.mapping.actor_dim):
            raise ValueError("Unexpected vector aircraft observation shape")

    def reset(self):
        observation=self.venv.reset();self._cache(observation)
        self.reset_infos=self.venv.reset_infos
        return observation

    def step_async(self,actions):
        if self._local is None:raise RuntimeError("Reset before choosing a maneuver")
        self.venv.step_async(self.mapping.decode(self._local,actions))

    def step_wait(self):
        result=self.venv.step_wait();self._cache(result[0]);self.reset_infos=self.venv.reset_infos
        return result


class ManeuverActor:
    """Choose a maneuver using one local observation and zero joint-critic inputs."""
    def __init__(self,model,mapping,configuration):
        if model.action_space!=mapping.action_space or model.observation_space["actor"].shape!=(mapping.actor_dim,):
            raise ValueError("Categorical policy and maneuver specification differ")
        if configuration.get("action_reference", "direct") != "direct":
            raise ValueError("Deployment must apply the maneuver mapping exactly once")
        expected_algorithm = "mappo" if model.policy.centralized else "ppo"
        if configuration.get("algorithm") != expected_algorithm:
            raise ValueError("Recorded algorithm differs from the policy critic")
        self.model = model
        self.mapping = mapping
        self.configuration = dict(configuration)
        self.observation_space = model.observation_space["actor"]
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float64)

    def __call__(self,observation):
        local=np.asarray(observation,dtype=np.float32)
        if local.shape!=(self.mapping.actor_dim,) or not np.isfinite(local).all():
            raise ValueError("Supply one finite local aircraft observation")
        inputs={"actor":local,"critic":np.zeros(self.model.observation_space["critic"].shape,dtype=np.float32)}
        choices=self.model.predict(inputs,deterministic=True)[0]
        return self.mapping.decode(local,choices)


def initialize_route_choice(model,probability=.9):
    """Explicit navigation prior; future learning must be compared to this same start."""
    import torch
    from stable_baselines3.common.distributions import MultiCategoricalDistribution
    if model.num_timesteps or model._n_updates or model.policy.optimizer.state:
        raise ValueError("Route-choice initialization requires a fresh model")
    if not np.isfinite(probability) or not .5<probability<1:
        raise ValueError("Preferred-category probability must lie strictly between .5 and 1")
    if not isinstance(model.policy.action_dist,MultiCategoricalDistribution) or model.action_space!=spaces.MultiDiscrete([20,3]):
        raise ValueError("Expected the registered categorical heading/speed policy")
    heading=np.full(20,(1-probability)/19);heading[0]=probability
    speed=np.full(3,(1-probability)/2);speed[2]=probability
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(torch.as_tensor(np.log(np.r_[heading,speed]),device=model.device,dtype=model.policy.action_net.bias.dtype))
