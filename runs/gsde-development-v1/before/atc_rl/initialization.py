"""Explicit, reproducible initialization controls for policy experiments."""


def neutral_action_mean(model):
    """Start a fresh Gaussian actor at zero heading and speed increments.

    This is only a weight initialization. No controller replaces the learned
    action during training or evaluation; every subsequent mean is learned.
    """
    import torch
    from gymnasium.spaces import Box
    if model.num_timesteps!=0 or model._n_updates!=0 or model.policy.optimizer.state:
        raise ValueError('Neutral initialization requires an untrained model')
    head=model.policy.action_net
    if not isinstance(model.action_space,Box) or not isinstance(head,torch.nn.Linear) or model.use_sde:
        raise ValueError('Expected a continuous diagonal-Gaussian actor')
    with torch.no_grad():
        head.weight.zero_()
        if head.bias is not None:head.bias.zero_()
