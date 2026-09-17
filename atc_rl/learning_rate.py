"""Validate the constant PPO step size and its serialized optimizer state."""
import math
from numbers import Real

DEFAULT = 3e-4


def constant_learning_rate(config):
    value = config.get("learning_rate", DEFAULT)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("Learning rate must be finite and positive")
    return float(value)


def validate_learning_rate(model, config):
    expected = constant_learning_rate(config)
    if callable(model.learning_rate) or model.learning_rate != expected:
        raise ValueError("Serialized learning rate differs from configuration")
    if any(model.lr_schedule(point) != expected for point in (0., .5, 1.)):
        raise ValueError("Serialized learning-rate schedule is not the recorded constant")
    if not model.policy.optimizer.param_groups or any(group["lr"] != expected for group in model.policy.optimizer.param_groups):
        raise ValueError("Serialized optimizer learning rate differs from configuration")
    return expected
