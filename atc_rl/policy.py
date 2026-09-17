"""PPO policies with explicit separation of actor and critic information."""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AircraftFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        self.actor_dim=int(np.prod(observation_space['actor'].shape))
        self.context_dim=int(np.prod(observation_space['critic'].shape))
        super().__init__(observation_space,self.actor_dim+self.context_dim)

    def forward(self, observations):
        return torch.cat((observations['actor'].flatten(1),observations['critic'].flatten(1)),dim=-1)

class SeparateInformation(nn.Module):
    def __init__(self, actor_dim, total_dim, centralized, actor_width=128, critic_width=256):
        super().__init__()
        self.actor_dim=actor_dim
        self.centralized=centralized
        self.latent_dim_pi=actor_width
        self.latent_dim_vf=critic_width
        self.policy_net=nn.Sequential(nn.Linear(actor_dim,actor_width),nn.Tanh(),nn.Linear(actor_width,actor_width),nn.Tanh())
        self.value_net=nn.Sequential(nn.Linear(total_dim if centralized else actor_dim,critic_width),nn.Tanh(),
                                     nn.Linear(critic_width,critic_width),nn.Tanh())

    def forward_actor(self, features):
        return self.policy_net(features[..., :self.actor_dim])

    def forward_critic(self, features):
        return self.value_net(features if self.centralized else features[..., :self.actor_dim])

    def forward(self, features):
        return self.forward_actor(features),self.forward_critic(features)

class AircraftPolicy(ActorCriticPolicy):
    """Shared actor; a local critic for PPO or a joint-information critic for MAPPO.

    Centralized inputs are concatenated only to simplify storage. The actor's
    computation explicitly slices them out. The MAPPO design follows centralized
    training with decentralized actors, as in Yu et al. (2022), arXiv:2103.01955.
    This is a feedforward implementation, not a reproduction of every setting
    in that paper. PPO optimization and distributions are supplied by SB3.
    """
    def __init__(self,*args,centralized=False,actor_width=128,critic_width=256,**kwargs):
        self.centralized=bool(centralized)
        self.actor_width=int(actor_width)
        self.critic_width=int(critic_width)
        if min(self.actor_width,self.critic_width)<1:raise ValueError('Network widths must be positive')
        kwargs['features_extractor_class']=AircraftFeatures
        kwargs['share_features_extractor']=True
        super().__init__(*args,**kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor=SeparateInformation(self.features_extractor.actor_dim,self.features_dim,
            self.centralized,self.actor_width,self.critic_width).to(self.device)

    def _get_constructor_parameters(self):
        result=super()._get_constructor_parameters()
        result.update(centralized=self.centralized,actor_width=self.actor_width,critic_width=self.critic_width)
        return result
