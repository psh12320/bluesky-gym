"""Read-only diagnostics over the same live samples used for PPO optimization."""
import numpy as np

FIELDS=('rollout_live_samples','rollout_return_variance_live','rollout_value_mse_live',
        'rollout_value_explained_variance_live','rollout_action_box_clip_fraction_live')


def rollout_diagnostics(buffer):
    """Use collected value predictions and GAE targets, before the policy update.

    Call after optimization has flattened the buffer. This never samples it or
    advances a random generator. Gaussian box clipping is distinct from heading
    mapping, route guidance and conflict-filter interventions.
    """
    if not buffer.full or not buffer.generator_ready:
        raise ValueError('Diagnostics require a complete, flattened rollout')
    mask=buffer.swap_and_flatten(buffer.valid).reshape(-1).astype(bool)
    indices=np.flatnonzero(mask)
    if not len(indices) or not np.array_equal(indices,buffer.valid_indices):
        raise ValueError('Live diagnostic samples differ from the optimizer mask')
    values=np.asarray(buffer.values).reshape(-1)
    returns=np.asarray(buffer.returns).reshape(-1)
    actions=np.asarray(buffer.actions)
    if len(values)!=len(mask) or len(returns)!=len(mask) or len(actions)!=len(mask):
        raise ValueError('Rollout diagnostic arrays have different sample counts')
    values=values[indices].astype(np.float64)
    returns=returns[indices].astype(np.float64)
    actions=actions[indices].reshape(len(indices),-1)
    if not all(np.isfinite(array).all() for array in (values,returns,actions)):
        raise ValueError('Non-finite live rollout diagnostic inputs')
    from gymnasium import spaces
    if isinstance(buffer.action_space,spaces.MultiDiscrete):
        low=buffer.action_space.start.reshape(1,-1)
        high=low+buffer.action_space.nvec.reshape(1,-1)
        if actions.shape[1]!=high.shape[1] or np.any(actions!=np.floor(actions)) or np.any(actions<low) or np.any(actions>=high):
            raise ValueError('Invalid categorical action indices in live rollout')
        clipping=None
    else:
        clipping=float(np.mean(np.any(np.abs(actions)>1.,axis=1)))
    variance=float(np.var(returns));errors=returns-values
    return dict(zip(FIELDS,(len(indices),variance,float(np.mean(errors**2)),
        float(1-np.var(errors)/variance) if variance>0 else None,
        clipping)))