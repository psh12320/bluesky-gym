import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from atc.algorithms import NavigationSAC
from atc_rl.sac_support import (LocalAircraftInputs, FiniteAircraftReplay,
                                SACAccounting, build_sac, verify_accounting)


class SyntheticAircraft(VecEnv):
    """An arrival followed by a deadline; the finished slot is padded once."""
    def __init__(self):
        self.phase = 0
        self.actions = None
        self.runtime = {"test": True}
        layout = spaces.Dict({"actor": spaces.Box(-10., 10., (3,), dtype=np.float32),
                              "critic": spaces.Box(-100., 100., (40,), dtype=np.float32)})
        super().__init__(10, layout, spaces.Box(-1., 1., (2,), dtype=np.float32))

    def observe(self, value):
        return {"actor": np.full((10, 3), value, dtype=np.float32),
                "critic": np.full((10, 40), 99., dtype=np.float32)}

    def reset(self):
        self.phase = 0
        self.reset_infos = [{"scenario_sha256": "reset"} for _ in range(10)]
        self._reset_seeds()
        return self.observe(0.)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        self.phase += 1
        final = self.phase == 2
        rewards = np.ones(10, dtype=np.float32)
        dones = np.ones(10, dtype=bool) if final else np.array([True] + [False]*9)
        infos = []
        for i in range(10):
            inactive = final and i == 0
            done = bool(dones[i]) and not inactive
            info = {"inactive": inactive, "aircraft_done": done,
                    "aircraft_terminated": done and not final,
                    "aircraft_truncated": done and final,
                    "TimeLimit.truncated": False, "world_completed": final and i == 0}
            if inactive:
                rewards[i] = 0.
            if done:
                info.update(metrics={"waypoint_reached": float(not final), "total_reward": 1.},
                            learning_episode_return=1., terminal_observation={
                                "actor": np.array([4., 5., 0.], dtype=np.float32),
                                "critic": np.full(40, 77., dtype=np.float32)})
            infos.append(info)
        self.last_infos = infos
        if final:
            self.phase = 0
        self.reset_infos = [{"scenario_sha256": "next" if final else ""} for _ in range(10)]
        return self.observe(0. if final else .1), rewards, dones, infos

    def close(self): pass
    def get_attr(self, name, indices=None): return [getattr(self, name, None) for _ in self._get_indices(indices)]
    def set_attr(self, name, value, indices=None): raise NotImplementedError
    def env_method(self, name, *args, indices=None, **kwargs): raise NotImplementedError
    def env_is_wrapped(self, wrapper_class, indices=None): return [False for _ in self._get_indices(indices)]


def test_local_inputs_remove_joint_state_and_transform_terminal_without_mutating_packet():
    original = SyntheticAircraft()
    env = LocalAircraftInputs(original)
    assert env.reset().shape == (10, 3)
    assert env.reset_infos == original.reset_infos
    actions = np.zeros((10, 2), dtype=np.float32)
    observations, rewards, dones, infos = env.step(actions)
    assert original.actions is actions
    assert observations.shape == (10, 3)
    np.testing.assert_array_equal(infos[0]["terminal_observation"], [4., 5., 0.])
    assert isinstance(original.last_infos[0]["terminal_observation"], dict)
    infos[0]["terminal_observation"][0] = -3.
    assert original.last_infos[0]["terminal_observation"]["actor"][0] == 4.
    assert rewards[0] == 1. and dones[0]


def test_fresh_sac_trains_on_all_live_steps_and_keeps_deadline_terminal_after_reload(tmp_path):
    torch.set_num_threads(1)
    model = build_sac(LocalAircraftInputs(SyntheticAircraft()), seed=26, device="cpu",
                      buffer_size=64, learning_starts=0, batch_size=8)
    inputs = np.ones((10, 3), dtype=np.float32)
    np.testing.assert_array_equal(model.predict(inputs, deterministic=True)[0], np.zeros((10, 2)))
    initial = {k: v.clone() for k, v in model.policy.state_dict().items()}
    account = SACAccounting()
    model.learn(40, callback=account)
    result = verify_accounting(model, account)
    assert result == {"live_transitions": 38, "padded_transitions": 2,
                      "counted_transitions": 40, "replay_size": 38}
    assert account.completed_worlds == 2 and len(account.completed_aircraft) == 20
    replay = model.replay_buffer
    # Continuing transitions bootstrap; arrival and deadline transitions do not.
    samples = replay._get_samples(np.arange(38))
    assert samples.dones.sum().item() == 20.
    assert samples.dones[1:10].sum().item() == 0.
    np.testing.assert_array_equal(replay.next_observations[0, 0], [4., 5., 0.])
    np.testing.assert_array_equal(replay.next_observations[10, 0], [4., 5., 0.])
    assert not replay.timeouts.any() and not replay.handle_timeout_termination
    changed = [k for k, v in model.policy.state_dict().items() if not torch.equal(v, initial[k])]
    assert any(k.startswith("actor.") for k in changed)
    assert any(k.startswith("critic.") for k in changed)
    assert model._n_updates == 4
    path = tmp_path / "model.zip"
    model.save(path)
    restored = NavigationSAC.load(path, device=model.device)
    np.testing.assert_array_equal(model.predict(inputs, deterministic=True)[0],
                                  restored.predict(inputs, deterministic=True)[0])
    assert restored.replay_buffer_class is FiniteAircraftReplay
    assert not restored.replay_buffer.handle_timeout_termination
    assert restored.gamma == .996508469331006


def test_replay_cannot_silently_restore_legacy_timeout_semantics():
    with pytest.raises(ValueError, match="deadline"):
        FiniteAircraftReplay(8, spaces.Box(-1., 1., (3,)), spaces.Box(-1., 1., (2,)),
                             handle_timeout_termination=True)


@pytest.mark.parametrize("problem", ["missing_activity", "missing_terminal", "inconsistent_terminal"])
def test_invalid_terminal_packets_fail_before_inserting_experience(problem):
    env = LocalAircraftInputs(SyntheticAircraft())
    replay = FiniteAircraftReplay(16, env.observation_space, env.action_space, device="cpu", n_envs=10)
    obs = env.reset()
    actions = np.zeros((10, 2), dtype=np.float32)
    next_obs, reward, done, infos = env.step(actions)
    if problem == "missing_activity": del infos[0]["inactive"]
    elif problem == "missing_terminal": del infos[0]["aircraft_terminated"]
    else: infos[0]["aircraft_terminated"] = False
    with pytest.raises(ValueError): replay.add(obs, next_obs, actions, reward, done, infos)
    assert replay.size() == replay.live_transitions == 0


def test_accounting_detects_collected_but_unstored_experience():
    model = build_sac(LocalAircraftInputs(SyntheticAircraft()), seed=28,
                      buffer_size=32, learning_starts=100, batch_size=8)
    account = SACAccounting()
    account.live_transitions = 1
    with pytest.raises(ValueError, match="counts disagree"):
        verify_accounting(model, account)


def test_inference_loader_and_native_actor_keep_only_local_inputs(tmp_path):
    from atc_rl.sac_support import load_sac_for_inference
    from atc_rl.deployment import LocalSACActor
    model = build_sac(LocalAircraftInputs(SyntheticAircraft()), seed=30, buffer_size=32)
    path = tmp_path / "initial.zip"
    model.save(path)
    config = {"algorithm": "sac", "gamma": model.gamma, "initial_action_std": .05,
              "actor_widths": [128, 128], "critic_widths": [256, 256], "number_of_critics": 2}
    restored = load_sac_for_inference(path, config)
    assert restored.buffer_size == 1
    actor = LocalSACActor(restored, config, {})
    np.testing.assert_array_equal(actor([1., 2., 3.]), [0., 0.])
    for malformed in (np.zeros((10, 3)), [1., np.nan, 3.]):
        with pytest.raises(ValueError): actor(malformed)
    with pytest.raises(ValueError, match="architecture"):
        load_sac_for_inference(path, {**config, "actor_widths": [64, 64]})


@pytest.mark.parametrize("option,value", [("--reward-scale", "nan"), ("--learning-starts", "-1"),
    ("--live-steps", "0"), ("--seed", "20301"), ("--initial-action-std", "1.1")])
def test_sac_invalid_configuration_fails_before_writing_run(tmp_path, option, value):
    import subprocess
    import sys
    from pathlib import Path
    folder = tmp_path / "rejected"
    result = subprocess.run([sys.executable, "-m", "atc_rl.train_sac", "--run-dir", str(folder), option, value],
                            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    assert result.returncode != 0 and not folder.exists()
