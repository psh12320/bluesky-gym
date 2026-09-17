import math

import numpy as np
import pytest

from atc.heading_transport import DecimalHeadingAction, decimal_heading
from core.actions import HeadingAction


@pytest.fixture(scope='module')
def simulator():
    # Initialize settings before the simulator's geography/parser modules load.
    from atc.envs import make_env
    env = make_env('ma')
    yield env
    env.close()


def test_actual_rejected_heading_roundtrips_through_simulator_parser(simulator):
    from bluesky.tools.misc import txt2hdg
    value = 7.843909918392455e-05
    with pytest.raises(ValueError, match='txt2hdg'):
        txt2hdg(str(value))
    text = decimal_heading(value)
    assert text == '0.00007843909918392455'
    assert txt2hdg(text) == value


def test_boundary_and_random_headings_preserve_exact_numeric_commands(simulator):
    from bluesky.tools.misc import txt2hdg
    rng = np.random.default_rng(7401)
    values = np.r_[[-180., 180., -0., 0., 7.843909918392455e-05],
                   np.nextafter(0., 1.), np.nextafter(0., -1.),
                   np.geomspace(1e-300, 1e-4, 64), -np.geomspace(1e-300, 1e-4, 64),
                   rng.uniform(-180, 180, 2000)]
    for value in values:
        text = decimal_heading(value)
        assert 'e' not in text.lower()
        assert float(text) == float(value)
        assert txt2hdg(text) == float(value)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_invalid_headings_are_rejected_before_queuing(value):
    with pytest.raises(ValueError, match='finite'):
        decimal_heading(value)


@pytest.mark.parametrize('n_discrete,actions', [(None, [-1., -.00001, 0., .00001, 1.]),
                                              (3, list(range(7)))])
def test_transport_preserves_original_continuous_and_discrete_targets(simulator, monkeypatch, n_discrete, actions):
    import bluesky as bs
    simulator.reset(seed=2026)
    captured=[]
    monkeypatch.setattr(bs.stack, 'stack', lambda command: captured.append(command))
    old = HeadingAction(45., n_discrete)
    new = DecimalHeadingAction(45., n_discrete)
    for action in actions:
        old.execute('KL001', action)
        new.execute('KL001', action)
        a, b = [line.split() for line in captured[-2:]]
        assert a[:2] == b[:2] == ['HDG', 'KL001']
        assert float(a[2]) == float(b[2])
        assert 'e' not in b[2].lower()
