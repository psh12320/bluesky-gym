from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from atc_rl.demonstrations import (
    CaptureAction, load_demonstrations, require_disjoint, validate_world,
)


def valid_arrays():
    return {
        "actor": np.tile(np.array([[.2, .3, 1.]], dtype=np.float32), (20, 1)),
        "teacher_action": np.zeros((20, 2), dtype=np.float32),
        "nominal_action": np.ones((20, 2), dtype=np.float32),
        "intervened": np.ones(20, dtype=bool),
        "native_reward": np.arange(20, dtype=float),
        "terminated": np.repeat([False, True], 10),
        "truncated": np.zeros(20, dtype=bool),
        "aircraft_index": np.tile(np.arange(10, dtype=np.int32), 2),
        "decision": np.repeat(np.arange(2, dtype=np.int32), 10),
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(directory, role="train", seed=61200, scenario="a"):
    directory.mkdir()
    plan = {"role": role, "seed": seed, "worlds": 1, "reference": None}
    (directory / "protocol.json").write_text(json.dumps(plan))
    np.savez(directory / "space.npz", actor_low=np.zeros(3), actor_high=np.ones(3))
    np.savez_compressed(directory / "world-00000.npz", **valid_arrays())
    worlds = [{"episode": 0, "file": "world-00000.npz",
               "sha256": digest(directory / "world-00000.npz"),
               "scenario_sha256": scenario, "live_transitions": 20}]
    (directory / "worlds.json").write_text(json.dumps(worlds))
    complete = {
        "status": "complete", "eligible_for_learning": role != "parity", "worlds": 1,
        "live_transitions": 20, "protocol_sha256": digest(directory / "protocol.json"),
        "space_sha256": digest(directory / "space.npz"),
        "world_manifest_sha256": digest(directory / "worlds.json"),
    }
    (directory / "complete.json").write_text(json.dumps(complete))
    return directory


def test_capture_delegates_original_value_once_and_preserves_attributes():
    calls = []
    class Action:
        d_speed = 10
        def execute(self, aircraft, value):
            calls.append((aircraft, value))
            return "sent"
    tap = CaptureAction(Action())
    original = np.float64(.123456789123)
    assert tap.execute("KL001", original) == "sent"
    assert calls[0][1] is original
    assert tap.values == {"KL001": float(original)}
    assert tap.d_speed == 10
    with pytest.raises(ValueError, match="Multiple"):
        tap.execute("KL001", original)
    assert len(calls) == 1
    tap.clear()
    tap.execute("KL001", -1.)
    assert len(calls) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.01, -1.01])
def test_capture_rejects_invalid_commands_without_execution(value):
    class Action:
        def execute(self, *args):
            raise AssertionError("Invalid command reached simulator")
    with pytest.raises(ValueError, match="finite and normalized"):
        CaptureAction(Action()).execute("KL001", value)


def test_validate_real_population_terminal_layout():
    arrays = valid_arrays()
    assert validate_world(arrays, 3) == 20
    arrays["terminated"][13] = False
    arrays["truncated"][13] = True
    assert validate_world(arrays, 3) == 20


@pytest.mark.parametrize("problem", ["duplicate", "missing_terminal", "early_terminal", "gap", "nan", "action"])
def test_invalid_or_inactive_training_samples_are_rejected(problem):
    arrays = valid_arrays()
    if problem == "duplicate":
        arrays["decision"][10] = 0
    elif problem == "missing_terminal":
        arrays["terminated"][10] = False
    elif problem == "early_terminal":
        arrays["terminated"][0] = True
    elif problem == "gap":
        arrays["decision"][10] = 4
    elif problem == "nan":
        arrays["actor"][0, 0] = np.nan
    elif problem == "action":
        arrays["teacher_action"][0, 0] = 1.1
    with pytest.raises(ValueError):
        validate_world(arrays, 3)


def test_complete_dataset_load_and_world_split(tmp_path):
    train = load_demonstrations(dataset(tmp_path / "train"), "train")
    validation = load_demonstrations(
        dataset(tmp_path / "val", role="validation", seed=61300, scenario="b"), "validation")
    assert train["arrays"]["actor"].shape == (20, 3)
    np.testing.assert_array_equal(train["world_index"], np.zeros(20))
    require_disjoint(train, validation)
    validation["scenarios"] = train["scenarios"]
    with pytest.raises(ValueError, match="same world"):
        require_disjoint(train, validation)


@pytest.mark.parametrize("role,seed", [("parity", 20260), ("train", 20260), ("train", 42), ("train", 20301)])
def test_evaluation_data_never_enter_learning(tmp_path, role, seed):
    path = dataset(tmp_path / "data", role=role, seed=seed)
    with pytest.raises(ValueError):
        load_demonstrations(path, "train")


def test_training_cannot_use_validation_role(tmp_path):
    path = dataset(tmp_path / "data", role="validation")
    with pytest.raises(ValueError, match="role"):
        load_demonstrations(path, "train")


def test_changed_dataset_archive_rejected(tmp_path):
    path = dataset(tmp_path / "data")
    with (path / "world-00000.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_demonstrations(path, "train")


def test_equal_generator_seed_rejected_even_with_different_worlds(tmp_path):
    train = load_demonstrations(dataset(tmp_path / "train"), "train")
    validation = load_demonstrations(
        dataset(tmp_path / "val", role="validation", scenario="b"), "validation")
    with pytest.raises(ValueError, match="seeds overlap"):
        require_disjoint(train, validation)
