"""Horizontal constant-velocity conflict features for a learned policy."""

import numpy as np
from gymnasium import spaces

HORIZON_SECONDS = 180.0
FEATURE_NAMES = ("traffic_present", "traffic_tcpa", "traffic_dcpa",
                 "traffic_entry_time", "traffic_predicted_conflict")


def conflict_features(position_m, velocity_mps, present, separation_m=9260.0,
                      horizon=HORIZON_SECONDS):
    """Predict closest approach and separation entry from current relative state.

    Predictions assume constant velocity over a finite horizon. They do not use
    future simulator state, alter the scorer, or guarantee future separation.
    Absent traffic slots are zeroed explicitly, including coincident positions.
    """
    position = np.asarray(position_m, dtype=float)
    velocity = np.asarray(velocity_mps, dtype=float)
    present = np.asarray(present, dtype=bool)
    if position.shape != velocity.shape or position.shape != (len(present), 2):
        raise ValueError("Relative positions/velocities must have shape (slots, 2)")
    if separation_m <= 0 or horizon <= 0:
        raise ValueError("Separation and prediction horizon must be positive")
    p2 = np.sum(position * position, axis=1)
    v2 = np.sum(velocity * velocity, axis=1)
    pv = np.sum(position * velocity, axis=1)
    moving = v2 > 1e-8
    closest_time = np.divide(-pv, v2, out=np.zeros_like(pv), where=moving)
    closest_time = np.clip(closest_time, 0.0, horizon)
    closest_distance = np.linalg.norm(position + closest_time[:, None] * velocity, axis=1)

    discriminant = pv * pv - v2 * (p2 - separation_m * separation_m)
    entry = np.divide(-pv - np.sqrt(np.maximum(discriminant, 0.0)), v2,
                      out=np.full_like(pv, horizon), where=moving)
    inside = p2 < separation_m * separation_m
    predicted = inside | (moving & (discriminant > 0) & (entry >= 0) & (entry < horizon))
    entry = np.where(inside, 0.0, np.where(predicted, entry, horizon))
    values = (present.astype(float), closest_time / horizon,
              np.clip(closest_distance / separation_m, 0.0, 4.0), entry / horizon,
              predicted.astype(float))
    return {name: np.where(present, value, 0.0) for name, value in zip(FEATURE_NAMES, values)}


class ConflictPrediction:
    """Append anticipatory features in the same slots as the original traffic state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        count = self.intruder_obs.n
        extra = {name: spaces.Box(0.0, 4.0 if name == "traffic_dcpa" else 1.0,
                                  (count,), dtype=np.float64) for name in FEATURE_NAMES}
        if hasattr(self, "observation_spaces"):
            self.observation_spaces = {agent: spaces.Dict({**space.spaces, **extra})
                                       for agent, space in self.observation_spaces.items()}
        else:
            self.observation_space = spaces.Dict({**self.observation_space.spaces, **extra})

    def _get_obs(self, ac_id):
        import bluesky as bs
        observation = super()._get_obs(ac_id)
        position = np.column_stack((observation["x_r"], observation["y_r"])) * self.intruder_obs.pos_norm
        velocity = np.column_stack((observation["vx_r"], observation["vy_r"])) * self.intruder_obs.spd_norm
        present = np.arange(self.intruder_obs.n) < max(bs.traf.ntraf - 1, 0)
        observation.update(conflict_features(position, velocity, present,
                                             separation_m=self.intrusion_distance * 1852.0))
        return observation
