import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box

from atc.algorithms import NavigationSAC, initialize_navigation_actor


class ControlSpace(Env):
    observation_space = Box(-1.0, 1.0, (4,), dtype=np.float32)
    action_space = Box(-1.0, 1.0, (2,), dtype=np.float32)


def test_navigation_actor_starts_at_zero_control_for_all_observations():
    model = NavigationSAC("MlpPolicy", ControlSpace(), buffer_size=10,
                          policy_kwargs={"net_arch": [8, 8]}, seed=17)
    initialize_navigation_actor(model)
    observations = np.random.default_rng(17).uniform(-1, 1, (20, 4)).astype(np.float32)
    assert np.all(model.predict(observations, deterministic=True)[0] == 0)
    action, replay_action = model._sample_action(100, n_envs=20)
    assert action.shape == (20, 2)
    np.testing.assert_allclose(action, replay_action, atol=1e-7, rtol=0)
    assert np.abs(action).max() < 0.6
