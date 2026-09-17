"""Explicit, reproducible initialization controls for policy experiments."""


def neutral_action_mean(model):
    """Start a fresh Gaussian or gSDE actor at zero heading and speed increments.

    This is only a weight initialization. No controller replaces the learned
    action during training or evaluation; every subsequent mean is learned.
    """
    import torch
    from gymnasium.spaces import Box
    if model.num_timesteps!=0 or model._n_updates!=0 or model.policy.optimizer.state:
        raise ValueError('Neutral initialization requires an untrained model')
    head=model.policy.action_net
    from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution
    if (not isinstance(model.action_space,Box) or not isinstance(head,torch.nn.Linear) or
            not isinstance(model.policy.action_dist,(DiagGaussianDistribution,StateDependentNoiseDistribution))):
        raise ValueError('Expected a continuous Gaussian or gSDE actor')
    with torch.no_grad():
        head.weight.zero_()
        if head.bias is not None:head.bias.zero_()
