import math

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from gymnasium import spaces

from atc_rl.imitation import (
    ObservationOnlyEnvironment, actor_mean, actor_parameters, fit_epoch,
    intervention_weights,
)
from atc_rl.initialization import neutral_action_mean
from atc_rl.policy import AircraftPolicy


@pytest.fixture
def model():
    torch.set_num_threads(1)
    environment = ObservationOnlyEnvironment(np.array([-1., -1., 0.]), np.ones(3))
    result = PPO(AircraftPolicy, environment, seed=82, device="cpu", n_steps=2, batch_size=2,
                 policy_kwargs={"log_std_init": math.log(.05), "actor_width": 32, "critic_width": 32})
    neutral_action_mean(result)
    return result


def test_observation_only_environment_cannot_generate_rl_experience():
    environment = ObservationOnlyEnvironment(np.zeros(3), np.ones(3))
    with pytest.raises(RuntimeError, match="RL experience"):
        environment.reset()
    with pytest.raises(RuntimeError, match="RL experience"):
        environment.step(np.zeros(2))


def test_supervised_action_space_matches_native_heading_and_speed():
    from core.actions import HeadingAction, SpeedAction, combine_action_spaces
    native = combine_action_spaces([HeadingAction(45), SpeedAction(20)])
    supervised = ObservationOnlyEnvironment(np.zeros(3), np.ones(3)).action_space
    assert supervised == native
    assert supervised.dtype == np.float64


@pytest.mark.parametrize("parent_dtype", [np.float32, np.float64])
def test_normalized_action_space_allows_equivalent_floating_storage(parent_dtype):
    from atc_rl.pretrained import action_space_compatibility
    parent = spaces.Box(-1, 1, shape=(2,), dtype=parent_dtype)
    target = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)
    result = action_space_compatibility(parent, target)
    assert result["parent_dtype"] == np.dtype(parent_dtype).name
    assert result["ppo_dtype"] == "float64"
    assert result["only_floating_storage_dtype_may_differ"]


@pytest.mark.parametrize("invalid", [
    spaces.Box(-2, 1, shape=(2,), dtype=np.float32),
    spaces.Box(-1, 2, shape=(2,), dtype=np.float64),
    spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
    spaces.Box(-1, 1, shape=(2,), dtype=np.int32),
    spaces.Box(-1, 1, shape=(2,), dtype=np.float16),
    spaces.Discrete(2),
])
def test_action_space_compatibility_rejects_changed_bounds_layout_or_type(invalid):
    from atc_rl.pretrained import action_space_compatibility
    valid = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)
    for parent, target in ((invalid, valid), (valid, invalid)):
        with pytest.raises(ValueError, match="normalized floating heading/speed"):
            action_space_compatibility(parent, target)


def test_intervention_objective_mass_is_explicit():
    flags = np.array([True] * 10 + [False] * 90)
    weights = intervention_weights(flags, .5)
    assert weights.sum() == pytest.approx(100)
    assert weights[flags].sum() == pytest.approx(50)
    assert weights[~flags].sum() == pytest.approx(50)
    np.testing.assert_array_equal(intervention_weights(np.zeros(8, dtype=bool), .5), np.ones(8))
    for fraction in (0., 1., float("nan")):
        with pytest.raises(ValueError):
            intervention_weights(flags, fraction)


def test_actor_fits_known_mapping_without_changing_critic_noise_or_rl_counters(model, tmp_path):
    rng = np.random.default_rng(84)
    observations = torch.tensor(rng.uniform(-1, 1, size=(128, 3)), dtype=torch.float32)
    observations[:, -1] = (observations[:, -1] + 1) / 2
    targets = torch.column_stack((.4 * observations[:, 0], -.3 * observations[:, 1]))
    weights = torch.ones(128)
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    optimizer = torch.optim.Adam(actor_parameters(model.policy), lr=.01)
    initial_error = float((actor_mean(model.policy, observations) - targets).square().mean().detach())
    for _ in range(25):
        loss, steps = fit_epoch(model.policy, observations, targets, weights, optimizer, rng.permutation(128), 32)
        assert steps == 4
    final_error = float((actor_mean(model.policy, observations) - targets).square().mean().detach())
    assert final_error < initial_error * .1
    changed = [k for k, v in model.policy.state_dict().items() if not torch.equal(before[k], v)]
    assert changed and all(k.startswith(("mlp_extractor.policy_net.", "action_net.")) for k in changed)
    assert model.num_timesteps == model._n_updates == 0
    assert not model.policy.optimizer.state
    obs = {"actor": observations.numpy(),
           "critic": rng.normal(size=(128, model.observation_space["critic"].shape[0])).astype(np.float32)}
    np.testing.assert_allclose(model.predict(obs, deterministic=True)[0],
                               actor_mean(model.policy, observations).detach().clamp(-1, 1).numpy(),
                               rtol=0, atol=1e-7)
    first = model.predict(obs, deterministic=True)[0]
    obs["critic"] *= 100
    np.testing.assert_array_equal(first, model.predict(obs, deterministic=True)[0])
    path = tmp_path / "bc.zip"
    model.save(path)
    restored = PPO.load(path, device=model.device)
    np.testing.assert_array_equal(first, restored.predict(obs, deterministic=True)[0])


def test_partial_or_repeated_epoch_is_rejected(model):
    observations = torch.zeros((8, 3))
    targets = torch.zeros((8, 2))
    optimizer = torch.optim.Adam(actor_parameters(model.policy))
    for permutation in (np.zeros(8, dtype=int), np.arange(7)):
        with pytest.raises(ValueError, match="exactly once"):
            fit_epoch(model.policy, observations, targets, torch.ones(8), optimizer, permutation, 4)


def test_nonfinite_fit_loss_is_rejected(model):
    optimizer = torch.optim.Adam(actor_parameters(model.policy))
    targets = torch.full((8, 2), float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        fit_epoch(model.policy, torch.zeros((8, 3)), targets, torch.ones(8), optimizer, np.arange(8), 4)


@pytest.mark.parametrize("parent_dtype", [np.float32, np.float64])
def test_complete_supervised_entrypoint_records_no_rl_updates(tmp_path, monkeypatch, parent_dtype):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    from atc_rl.imitation import main
    from atc_rl.demonstrations import sha256
    helper_path = Path(__file__).with_name("test_demonstrations.py")
    spec = importlib.util.spec_from_file_location("demonstration_test_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    for label, role, seed, scenario in (("train", "train", 61200, "a"), ("val", "validation", 61300, "b")):
        directory = helper.dataset(tmp_path / label, role=role, seed=seed, scenario=scenario)
        protocol = json.loads((directory / "protocol.json").read_text())
        protocol["source_sha256"] = {}
        protocol["synthetic_test_fixture"] = True
        (directory / "protocol.json").write_text(json.dumps(protocol))
        arrays = helper.valid_arrays()
        arrays["teacher_action"][:] = [.3, -.2]
        np.savez_compressed(directory / "world-00000.npz", **arrays)
        worlds = json.loads((directory / "worlds.json").read_text())
        worlds[0]["sha256"] = sha256(directory / "world-00000.npz")
        (directory / "worlds.json").write_text(json.dumps(worlds))
        complete = json.loads((directory / "complete.json").read_text())
        complete["protocol_sha256"] = sha256(directory / "protocol.json")
        complete["world_manifest_sha256"] = sha256(directory / "worlds.json")
        (directory / "complete.json").write_text(json.dumps(complete))
    output = tmp_path / "fit"
    monkeypatch.setattr(sys, "argv", [
        "imitation", "--training-data", str(tmp_path / "train"), "--validation-data", str(tmp_path / "val"),
        "--out", str(output), "--epochs", "3", "--batch-size", "8",
    ])
    main()
    summary = json.loads((output / "training_summary.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert config["algorithm"] == "behavior_cloning"
    assert not config["reinforcement_learning_performed"]
    assert summary["supervised_optimizer_steps"] == 9
    assert summary["supervised_examples_seen"] == 60
    assert summary["reinforcement_learning_updates"] == 0
    assert summary["critic_and_noise_unchanged"] and summary["serialization_actions_identical"]
    assert summary["final_errors"]["validation"]["action_mse"] < summary["initial_errors"]["validation"]["action_mse"]
    checkpoints = json.loads((output / "checkpoints.json").read_text())
    assert all(r["live_transitions"] == r["optimizer_steps"] == 0 for r in checkpoints)
    assert checkpoints[-1]["supervised_optimizer_steps"] == 9
    from atc_rl.checkpoint_identity import policy_fingerprint
    parent = PPO.load(output / "model.zip", device="cpu")
    original_fingerprint = policy_fingerprint(output / "model.zip")
    # Reproduce the legacy metadata only in this synthetic temporary fixture.
    parent.action_space = spaces.Box(-1, 1, shape=(2,), dtype=parent_dtype)
    parent.policy.action_space = parent.action_space
    parent.save(output / "model.zip")
    assert policy_fingerprint(output / "model.zip") == original_fingerprint
    checkpoints[-1]["sha256"] = sha256(output / "model.zip")
    summary["model_sha256"] = checkpoints[-1]["sha256"]
    (output / "checkpoints.json").write_text(json.dumps(checkpoints))
    (output / "training_summary.json").write_text(json.dumps(summary))
    from atc_rl.pretrained import initialize_from_pretrained, audit_pretraining
    import random
    fresh = PPO(AircraftPolicy, ObservationOnlyEnvironment(np.zeros(3), np.ones(3)),
                seed=115, device="cpu", n_steps=2, batch_size=2,
                policy_kwargs={"log_std_init": math.log(.05)})
    rl_config = {**config, "algorithm": "ppo", "neutral_action_mean": False,
                 "pretrained_model": str(output / "model.zip")}
    warm = tmp_path / "warm"
    warm.mkdir()
    torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    lineage = initialize_from_pretrained(fresh, output / "model.zip", rl_config, warm)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert numpy_rng[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
    assert numpy_rng[2:] == np.random.get_state()[2:]
    assert python_rng == random.getstate()
    assert not fresh.policy.optimizer.state and fresh.num_timesteps == fresh._n_updates == 0
    assert fresh.action_space.dtype == np.float64
    assert fresh.policy.action_space.dtype == np.float64
    assert lineage["action_space_compatibility"]["parent_dtype"] == np.dtype(parent_dtype).name
    observations = {
        "actor": np.linspace(0, 1, 18, dtype=np.float32).reshape(6, 3),
        "critic": np.zeros((6, fresh.observation_space["critic"].shape[0]), dtype=np.float32),
    }
    np.testing.assert_array_equal(parent.predict(observations, deterministic=True)[0],
                                  fresh.predict(observations, deterministic=True)[0])
    fresh.save(warm / "initial-model.zip")
    rl_config["pretraining"] = lineage
    assert audit_pretraining(warm, rl_config) == lineage
    assert policy_fingerprint(warm / "initial-model.zip") == original_fingerprint
    import copy
    wrong_lineage = copy.deepcopy(rl_config)
    wrong_lineage["pretraining"]["action_space_compatibility"]["ppo_dtype"] = "float32"
    with pytest.raises(ValueError, match="compatibility record"):
        audit_pretraining(warm, wrong_lineage)
    assert lineage["supervised_optimizer_steps"] == 9
    assert not lineage["initial_policy_is_untrained"]
    mismatched = {**rl_config, "action_reference": "goal_offset"}
    with pytest.raises(ValueError, match="action_reference"):
        initialize_from_pretrained(fresh, output / "model.zip", mismatched, tmp_path / "bad")
    archived = warm / "pretraining/config.json"
    archived.write_text(archived.read_text() + " ")
    with pytest.raises(ValueError, match="lineage changed"):
        audit_pretraining(warm, rl_config)

