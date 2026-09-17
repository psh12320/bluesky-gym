import subprocess
import sys
from pathlib import Path
import pytest
import torch
from stable_baselines3 import PPO
from atc_rl.learning_rate import constant_learning_rate, validate_learning_rate
from atc_rl.policy import AircraftPolicy
from tests.rl.test_exploration import ConstantObservation


@pytest.mark.parametrize("rate", [3e-4, 3e-5])
@pytest.mark.parametrize("centralized", [False, True])
def test_learning_rate_survives_training_and_cpu_reload(tmp_path, rate, centralized):
    torch.set_num_threads(1)
    model = PPO(AircraftPolicy, ConstantObservation(), seed=92, device="cpu", n_steps=4, batch_size=4,
                n_epochs=1, learning_rate=rate,
                policy_kwargs={"centralized": centralized, "actor_width": 8, "critic_width": 8})
    before = {name: value.clone() for name, value in model.policy.state_dict().items()}
    assert validate_learning_rate(model, {"learning_rate": rate}) == rate
    model.learn(8)
    assert any(not torch.equal(value, before[name]) for name, value in model.policy.state_dict().items())
    path = tmp_path / "model.zip"
    model.save(path)
    restored = PPO.load(path, device="cpu")
    assert validate_learning_rate(restored, {"learning_rate": rate}) == rate
    restored.policy.optimizer.param_groups[0]["lr"] = rate / 10
    with pytest.raises(ValueError, match="optimizer learning rate"):
        validate_learning_rate(restored, {"learning_rate": rate})


def test_default_remains_unchanged_and_bad_rates_are_rejected():
    assert constant_learning_rate({}) == 3e-4
    for value in (0., -1., float("nan"), float("inf"), True, "0.001"):
        with pytest.raises(ValueError, match="finite and positive"):
            constant_learning_rate({"learning_rate": value})


@pytest.mark.parametrize("value", ["0", "-0.001", "nan", "inf"])
def test_invalid_cli_rate_rejected_before_creating_run(tmp_path, value):
    directory = tmp_path / "rejected"
    result = subprocess.run([sys.executable, "-m", "atc_rl.train", "--learning-rate", value,
                             "--run-dir", str(directory)], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[2])
    assert result.returncode != 0 and "Learning rate must be finite and positive" in result.stderr
    assert not directory.exists()


def test_audit_rejects_changed_config_or_schedule():
    torch.set_num_threads(1)
    model = PPO(AircraftPolicy, ConstantObservation(), device="cpu", n_steps=2, batch_size=2,
                learning_rate=3e-5, policy_kwargs={"actor_width": 8, "critic_width": 8})
    with pytest.raises(ValueError, match="differs from configuration"):
        validate_learning_rate(model, {"learning_rate": 3e-4})
    model.lr_schedule = lambda point: 3e-5 * point
    with pytest.raises(ValueError, match="schedule"):
        validate_learning_rate(model, {"learning_rate": 3e-5})
