import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PredictionResidual(BaseFeaturesExtractor):
    """Add new observations through an initially zero projection.

    The original MLP keeps its input dimension and existing feature order. At
    initialization its input is unchanged, including its matrix/vector shape.
    The projection can be frozen at zero for a matched fine-tuning control.
    """

    def __init__(self, observation_space, original_indices, added_indices, enabled=True):
        super().__init__(observation_space, features_dim=len(original_indices))
        self.register_buffer("original_indices", torch.tensor(original_indices, dtype=torch.long))
        self.register_buffer("added_indices", torch.tensor(added_indices, dtype=torch.long))
        self.projection = nn.Linear(len(added_indices), len(original_indices), bias=False)
        nn.init.zeros_(self.projection.weight)
        self.projection.requires_grad_(enabled)

    def forward(self, observations):
        original = observations.index_select(1, self.original_indices)
        additional = observations.index_select(1, self.added_indices)
        return original + self.projection(additional)
