"""Serialize horizontal heading commands without exponent notation."""
from decimal import Decimal
import math

import bluesky as bs
from core.actions import HeadingAction, _bound_angle, _discrete_to_float

REVISION = 1


def decimal_heading(value):
    """Return decimal text that preserves the finite heading's float value."""
    if not math.isfinite(float(value)):
        raise ValueError('Heading must be finite')
    return format(Decimal(str(value)), 'f')


class DecimalHeadingAction(HeadingAction):
    """Retain the original heading calculation and change only its text format."""

    def execute(self, ac_id, action):
        if self.n_discrete is not None:
            action = _discrete_to_float(action, self.n_discrete)
        index = bs.traf.id2idx(ac_id)
        heading = _bound_angle(bs.traf.hdg[index] + float(action) * self.d_heading)
        bs.stack.stack(f'HDG {ac_id} {decimal_heading(heading)}')


def attach_decimal_heading(env):
    """Explicitly opt an existing environment into decimal command serialization."""
    world = env.unwrapped
    previous = world.heading_action
    if type(previous) is not HeadingAction:
        raise ValueError('Expected the original HeadingAction before attaching transport')
    world.heading_action = DecimalHeadingAction(previous.d_heading, previous.n_discrete)
    return env
