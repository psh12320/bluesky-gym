import numpy as np


def predict_actions(model, observations, batched=False):
    """Match the official per-aircraft call shape unless reproducing an old run.

    Matrix and vector inference can differ in floating-point rounding, and small
    action differences can grow over a long interacting-aircraft rollout.
    """
    observations = np.asarray(observations)
    if observations.ndim == 1 or batched:
        return model.predict(observations, deterministic=True)[0]
    return np.stack([model.predict(observation, deterministic=True)[0]
                     for observation in observations])
