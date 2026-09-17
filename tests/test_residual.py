from types import SimpleNamespace
import numpy as np
import pytest
from gymnasium import spaces

from atc.baselines import goal_tracker
from atc.residual import RouteResidualControl, compose_command, configuration, reference_turn
from atc.route_input import RouteInput, configuration as input_configuration
from atc.submission import _validate_configuration


@pytest.mark.parametrize('fast', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_zero_residual_exactly_matches_classical_reference(dtype, fast):
    dictionary = spaces.Dict({'cos_drift': spaces.Box(-1., 1., (1,), dtype=dtype),
                              'sin_drift': spaces.Box(-1., 1., (1,), dtype=dtype)})
    controller = goal_tracker(dictionary, speed_action=1.0 if fast else 0.0)
    for angle in np.r_[np.linspace(-np.pi, np.pi, 1001), -np.pi/4, np.pi/4, 0]:
        cosine, sine = dtype(np.cos(angle)), dtype(np.sin(angle))
        expected = controller(np.array([cosine, sine], dtype=dtype))
        actual = compose_command(reference_turn(cosine, sine), [0, 0], fast_speed_reference=fast)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float32


def test_residual_is_bounded_and_keeps_original_command_limits():
    for reference in np.linspace(-1, 1, 101, dtype=np.float32):
        for heading in (-10, -1, -.5, 0, .5, 1, 10):
            for speed in (-10, -.3, 0, .8, 10):
                command = compose_command(reference, [heading, speed])
                assert np.all(np.abs(command) <= 1)
                assert abs(float(command[0]) - float(reference)) * 45 <= 15 + 1e-5
                assert command[1] == pytest.approx(np.clip(speed, -1, 1))


def test_residual_dispatch_uses_the_last_observation_and_records_adjustment(monkeypatch):
    observations = iter([dict(cos_drift=np.array([1.]), sin_drift=np.array([0.])),
                         dict(cos_drift=np.array([0.]), sin_drift=np.array([1.]))])
    monkeypatch.setattr(RouteInput, '_get_obs', lambda self, ac_id: next(observations))
    received = []
    def dispatch(self, ac_id, action):
        received.append(action.copy())
        return 'inner'
    monkeypatch.setattr(RouteInput, '_get_action', dispatch)
    env = RouteResidualControl.__new__(RouteResidualControl)
    env._clear_residual_state()
    env._get_obs('AC')
    assert env._get_action('AC', [1, -.5]) == 'inner'
    np.testing.assert_allclose(received[-1], [1/3, -.5])
    env._get_obs('AC')
    env._get_action('AC', [0, 0])
    np.testing.assert_array_equal(received[-1], [-1, 0])
    assert env.residual_statistics['actions'] == 2
    assert env.residual_statistics['heading_adjusted_actions'] == 1
    assert env.residual_statistics['speed_adjusted_actions'] == 1


def test_reset_clears_cached_reference_and_statistics(monkeypatch):
    monkeypatch.setattr(RouteInput, 'reset', lambda self, *a, **k: 'reset')
    env = RouteResidualControl.__new__(RouteResidualControl)
    env._clear_residual_state()
    env._reference_turns['AC'] = .7
    env.residual_statistics['actions'] = 7
    assert env.reset() == 'reset'
    assert env._reference_turns == {}
    assert not any(env.residual_statistics.values())


def test_checkpoint_requires_matching_residual_mapping():
    config = dict(env='ma', algorithm='sac', recipe='public_route_residual',
                  **input_configuration(), **configuration())
    _validate_configuration('ma', config)
    config['route_residual_heading_limit_deg'] = 30
    with pytest.raises(ValueError, match='route-residual settings'):
        _validate_configuration('ma', config)


def test_navigation_initialization_predicts_exact_zero_adjustment():
    import gymnasium as gym
    import torch
    from atc.algorithms import NavigationSAC, initialize_navigation_actor
    class LayoutEnv(gym.Env):
        observation_space = spaces.Box(-np.inf, np.inf, (124,), dtype=np.float64)
        action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
    torch.set_num_threads(1)
    model = NavigationSAC('MlpPolicy', LayoutEnv(), buffer_size=10,
                          policy_kwargs={'net_arch': [16, 16]}, seed=2800)
    initialize_navigation_actor(model)
    observation = np.random.default_rng(2800).normal(size=(7, 124))
    action, _ = model.predict(observation, deterministic=True)
    np.testing.assert_array_equal(action, np.zeros((7, 2), dtype=np.float32))


def test_fast_reference_keeps_braking_authority_and_heading_limits():
    for reference in np.linspace(-1, 1, 101, dtype=np.float32):
        for heading in (-2, -1, 0, 1, 2):
            for residual, expected in ((-2, -1), (-1, -1), (-.75, -.5), (-.5, 0), (-.25, .5), (0, 1), (1, 1)):
                command = compose_command(reference, [heading, residual], fast_speed_reference=True)
                assert command[1] == expected
                assert abs(float(command[0]) - float(reference)) * 45 <= 15 + 1e-5
                assert np.all(np.abs(command) <= 1)


def test_fast_mapping_metadata_rejects_legacy_and_missing_reference():
    config = dict(env='ma', algorithm='sac', recipe='public_route_choice_fast_residual',
                  **input_configuration(True), **configuration(True))
    _validate_configuration('ma', config)
    del config['route_residual_speed_reference']
    with pytest.raises(ValueError, match='route-residual settings'):
        _validate_configuration('ma', config)
    config.update(configuration(False))
    with pytest.raises(ValueError, match='route-residual settings'):
        _validate_configuration('ma', config)
    assert 'route_residual_speed_reference' not in configuration()
    assert configuration()['route_residual_speed_scale'] == 1.0


def test_fast_dispatch_counts_adjustments_from_fast_reference(monkeypatch):
    received = []
    monkeypatch.setattr(RouteInput, '_get_action', lambda self, ac_id, action: received.append(action.copy()))
    env = RouteResidualControl.__new__(RouteResidualControl)
    env.fast_speed_reference = True
    env._clear_residual_state()
    env._reference_turns['AC'] = np.float32(.25)
    env._get_action('AC', [0, 0])
    np.testing.assert_array_equal(received[-1], [.25, 1])
    assert env.residual_statistics['speed_adjusted_actions'] == 0
    env._get_action('AC', [0, -.75])
    np.testing.assert_array_equal(received[-1], [.25, -.5])
    assert env.residual_statistics['speed_adjusted_actions'] == 1
    assert env.last_residual['AC']['nominal_speed_adjustment'] == -1.5
