import pytest

from atc.recipes import Recipe, RECIPES
from atc.route_input import configuration as route_configuration
from atc.residual import configuration as residual_configuration
from atc.submission import _validate_configuration


@pytest.mark.parametrize('interval', [0, -1, 1.5, True])
def test_invalid_decision_interval_rejected(interval):
    with pytest.raises(ValueError, match='positive integer'):
        Recipe(decision_interval_seconds=interval)


def test_interval_and_speed_are_independent_constructor_settings():
    assert Recipe().action_kwargs() == {}
    assert Recipe().action_configuration() == {}
    recipe = RECIPES['public_route_choice_fast_residual_interval5']
    assert recipe.action_kwargs() == {'action_frequency': 5}
    assert recipe.action_configuration() == {'decision_interval_seconds': 5}
    combined = Recipe(speed_increment_knots=20., decision_interval_seconds=5)
    assert combined.action_kwargs() == {'d_speed': 20., 'action_frequency': 5}
    assert combined.action_configuration() == {'speed_command_increment_knots': 20., 'decision_interval_seconds': 5}
    assert RECIPES['public_route_choice_fast_residual'].action_kwargs() == {}


def test_interval_checkpoint_requires_matching_saved_rate():
    config = dict(env='ma', algorithm='sac', recipe='public_route_choice_fast_residual_interval5',
                  **route_configuration(True), **residual_configuration(True))
    with pytest.raises(ValueError, match='decision-interval settings'):
        _validate_configuration('ma', config)
    config['decision_interval_seconds'] = 5
    _validate_configuration('ma', config)
    config['decision_interval_seconds'] = 10
    with pytest.raises(ValueError, match='decision-interval settings'):
        _validate_configuration('ma', config)
