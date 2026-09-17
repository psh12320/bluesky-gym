import numpy as np
import pytest
from shapely.geometry import box

from atc.recipes import Recipe, RECIPES
from atc.route_input import configuration as route_configuration
from atc.residual import configuration as residual_configuration
from atc.submission import _validate_configuration
from atc.traffic_projection import choose_command, predict_commands
from core.actions import MpS2Kt, SpeedAction


def test_larger_braking_command_prevents_overtake_in_a_narrow_corridor():
    other = predict_commands([10.5, 0], 90, 200, [0], [200])
    def assess(increment):
        target = lambda action: 210 + np.asarray(action) * increment / MpS2Kt
        return choose_command(box(-100, -.1, 100, .1), [0, 0], 90, 210,
                              [0, 0], target, other, [100])
    _, _, _, original = assess(20 / 3)
    chosen, _, _, larger = assess(20)
    assert original['no_jointly_feasible_candidate']
    assert not larger['no_jointly_feasible_candidate']
    assert chosen[0] == 0 and chosen[1] == -1
    assert larger['predicted_minimum_separation_km'] >= 10.26


def test_speed_command_uses_knots_and_leaves_aircraft_dynamics_to_simulator(monkeypatch):
    from types import SimpleNamespace
    import core.actions as actions
    commands = []
    monkeypatch.setattr(actions, 'bs', SimpleNamespace(
        traf=SimpleNamespace(id2idx=lambda _: 0, cas=[210.0]),
        stack=SimpleNamespace(stack=commands.append)))
    SpeedAction(**RECIPES['public_route_choice_speed20'].action_kwargs()).execute('AC', -1)
    assert len(commands) == 1
    prefix, agent, target = commands[0].split()
    assert (prefix, agent) == ('SPD', 'AC')
    assert float(target) == pytest.approx(210 * MpS2Kt - 20)


def test_speed_mapping_checkpoint_must_preserve_the_command_increment():
    recipe = RECIPES['public_route_choice_speed20_residual']
    config = dict(env='ma', algorithm='sac', recipe='public_route_choice_speed20_residual',
                  **route_configuration(True), **residual_configuration())
    with pytest.raises(ValueError, match='speed-command settings'):
        _validate_configuration('ma', config)
    config.update(recipe.action_configuration())
    _validate_configuration('ma', config)
    config['speed_command_increment_knots'] = 20 / 3
    with pytest.raises(ValueError, match='speed-command settings'):
        _validate_configuration('ma', config)
    assert RECIPES['public_route_choice_residual'].action_kwargs() == {}
    assert RECIPES['public_route_choice_residual'].action_configuration() == {}


@pytest.mark.parametrize('increment', [0, -1, float('nan'), float('inf')])
def test_invalid_speed_increment_is_rejected(increment):
    with pytest.raises(ValueError, match='finite and positive'):
        Recipe(speed_increment_knots=increment)
