"""Classical controls for measuring the contribution of a learned policy."""

from gymnasium import spaces
import numpy as np


def goal_tracker(observation_dictionary, speed_action=0.0):
    """Track the observed bearing with a constant normalized speed command."""
    if not np.isfinite(speed_action) or not -1 <= speed_action <= 1:
        raise ValueError("speed_action must be finite and within [-1, 1]")
    offsets, cursor = {}, 0
    for name, space in observation_dictionary.spaces.items():
        offsets[name] = cursor
        cursor += spaces.flatdim(space)
    cosine, sine = offsets["cos_drift"], offsets["sin_drift"]

    def predict(observations):
        values = np.asarray(observations)
        # Drift is own heading minus goal bearing. Direct turns are +/-45 deg.
        turn = np.clip(-np.arctan2(values[..., sine], values[..., cosine]) / (np.pi / 4), -1, 1)
        return np.stack((turn, np.full_like(turn, speed_action)), axis=-1).astype(np.float32)
    return predict
