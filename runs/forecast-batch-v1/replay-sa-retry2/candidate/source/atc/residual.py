"""Learn bounded changes to the verified route-following command."""
import numpy as np

from atc.route_input import RouteInput

REVISION = 1
HEADING_LIMIT_DEG = 15.0
DIRECT_TURN_LIMIT_DEG = 45.0


def configuration(fast_speed_reference=False):
    result = dict(route_residual_revision=REVISION,
                route_residual_heading_limit_deg=HEADING_LIMIT_DEG,
                route_residual_speed_scale=1.0,
                route_residual_composition='clipped_direct_reference_plus_adjustment',
                route_residual_reference='last_observation_bearing_float32')
    if fast_speed_reference:
        result.update(route_residual_speed_scale=2.0, route_residual_speed_reference=1.0)
    return result


def reference_turn(cosine, sine):
    """Match the classical goal tracker's command, including its output dtype."""
    return np.float32(np.clip(-np.arctan2(sine, cosine) / (np.pi / 4), -1, 1))


def compose_command(reference, action, fast_speed_reference=False):
    residual = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
    return np.array([np.clip(float(reference) + residual[0] * HEADING_LIMIT_DEG / DIRECT_TURN_LIMIT_DEG,
                             -1, 1),
                     np.clip(1 + 2 * residual[1], -1, 1) if fast_speed_reference else residual[1]],
                    dtype=np.float32)


class RouteResidualControl(RouteInput):
    """Use the cached observation bearing so zero action matches route tracking."""

    fast_speed_reference = False

    def __init__(self, *args, fast_speed_reference=False, **kwargs):
        self.fast_speed_reference = fast_speed_reference
        self._clear_residual_state()
        super().__init__(*args, **kwargs)

    def _clear_residual_state(self):
        self._reference_turns = {}
        self.last_residual = {}
        self.residual_statistics = dict(actions=0, heading_adjusted_actions=0,
            speed_adjusted_actions=0, absolute_heading_adjustment_deg=0.0,
            absolute_speed_action=0.0)

    def reset(self, *args, **kwargs):
        self._clear_residual_state()
        return super().reset(*args, **kwargs)

    def _get_obs(self, ac_id):
        observation = super()._get_obs(ac_id)
        self._reference_turns[ac_id] = reference_turn(observation['cos_drift'][0],
                                                    observation['sin_drift'][0])
        return observation

    def _get_action(self, ac_id, action):
        if ac_id not in self._reference_turns:
            self._get_obs(ac_id)
        reference = self._reference_turns[ac_id]
        command = compose_command(reference, action, self.fast_speed_reference)
        speed_reference = 1.0 if self.fast_speed_reference else 0.0
        adjustment = (float(command[0]) - float(reference)) * DIRECT_TURN_LIMIT_DEG
        self.residual_statistics['actions'] += 1
        self.residual_statistics['heading_adjusted_actions'] += int(command[0] != reference)
        self.residual_statistics['speed_adjusted_actions'] += int(command[1] != speed_reference)
        self.residual_statistics['absolute_heading_adjustment_deg'] += abs(adjustment)
        self.residual_statistics['absolute_speed_action'] += abs(float(command[1]))
        self.last_residual[ac_id] = dict(reference_heading_turn_deg=float(reference) * DIRECT_TURN_LIMIT_DEG,
            nominal_heading_adjustment_deg=adjustment, nominal_speed_action=float(command[1]))
        if self.fast_speed_reference:
            self.last_residual[ac_id].update(reference_speed_action=speed_reference,
                nominal_speed_adjustment=float(command[1]) - speed_reference)
        result = super()._get_action(ac_id, command)
        if hasattr(self, 'last_projection') and ac_id in self.last_projection:
            self.last_projection[ac_id].update(self.last_residual[ac_id])
        return result
