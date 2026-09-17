import numpy as np
from gymnasium.spaces import Box
from supersuit.vector import MarkovVectorEnv

from atc.replay import AircraftReplayBuffer
from atc.vector import FixedPopulation
from test_vector import TwoAircraft


def test_arrival_ends_return_immediately_and_padding_never_enters_replay():
    world = MarkovVectorEnv(FixedPopulation(TwoAircraft()))
    obs, _ = world.reset()
    action = np.zeros((2, 1), dtype=np.float32)
    buffer = AircraftReplayBuffer(8, Box(-1.0, 1.0, (1,)), Box(-1.0, 1.0, (1,)), n_envs=2)
    next_obs, rewards, terms, truncs, infos = world.step(action)
    buffer.add(obs, next_obs, action, rewards, terms | truncs, infos)
    assert buffer.pos == 2
    assert buffer.dones[:2, 0].tolist() == [1.0, 0.0]
    assert buffer.rewards[:2, 0].tolist() == [5.0, 0.0]
    obs = next_obs
    next_obs, rewards, terms, truncs, infos = world.step(action)
    # SB3 replaces the automatic reset observation with terminal_observation.
    for i, done in enumerate(terms | truncs):
        if done:
            next_obs[i] = infos[i]["terminal_observation"]
    buffer.add(obs, next_obs, action, rewards, terms | truncs, infos)
    assert buffer.pos == 3
    assert buffer.live_transitions == 3
    assert buffer.skipped_transitions == 1
    assert buffer.next_observations[2, 0].tolist() == [1.0]
    assert buffer.timeouts[:3, 0].tolist() == [0.0, 0.0, 1.0]
    sample = buffer._get_samples(np.array([0, 1, 2]))
    assert sample.dones.flatten().tolist() == [1.0, 0.0, 0.0]


def test_replay_ring_wrap_keeps_only_recent_live_transitions():
    buffer = AircraftReplayBuffer(2, Box(-10.0, 10.0, (1,)), Box(-1.0, 1.0, (1,)), n_envs=3)
    obs = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    buffer.add(obs, obs, np.zeros((3, 1)), np.arange(3), np.zeros(3), [{}, {}, {}])
    assert buffer.full and buffer.pos == 1
    assert buffer.observations[:, 0, 0].tolist() == [3.0, 2.0]
