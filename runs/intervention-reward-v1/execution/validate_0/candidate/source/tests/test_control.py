import numpy as np
import pytest

from atc.control import goal_relative_command


@pytest.mark.parametrize("heading,bearing,expected", [(350, 10, 20 / 45), (10, 350, -20 / 45), (180, 180, 0), (0, 100, 1), (180, 80, -1)])
def test_goal_relative_control_wraps_headings_and_respects_turn_limit(heading, bearing, expected):
    assert goal_relative_command(0.0, heading, bearing) == pytest.approx(expected)


def test_learned_offset_can_choose_either_side_of_goal():
    assert goal_relative_command(0.25, 0.0, 0.0) == 0.5
    assert goal_relative_command(-0.25, 0.0, 0.0) == -0.5
    commands = [goal_relative_command(a, h, 0) for a in np.linspace(-1, 1, 50) for h in range(360)]
    assert min(commands) >= -1 and max(commands) <= 1
