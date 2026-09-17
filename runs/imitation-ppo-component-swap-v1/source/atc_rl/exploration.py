"""Explicit Gaussian/gSDE exploration settings and checkpoint validation."""
import math


def configuration(config):
    mode = config.get("exploration", "gaussian")
    weight = config.get("sde_weight_std")
    frequency = config.get("sde_sample_freq")
    if mode == "gaussian":
        if weight is not None or frequency is not None:
            raise ValueError("gSDE settings require exploration=gsde")
    elif mode == "gsde":
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("gSDE weight standard deviation must be finite and in (0, 1]")
        if type(frequency) is not int or frequency < 1:
            raise ValueError("gSDE resampling requires a positive number of decisions")
    else:
        raise ValueError("Unknown exploration mode")
    return {"exploration": mode, "sde_weight_std": weight, "sde_sample_freq": frequency}


def ppo_options(config):
    settings = configuration(config)
    enabled = settings["exploration"] == "gsde"
    return {"use_sde": enabled, "sde_sample_freq": settings["sde_sample_freq"] if enabled else -1}


def log_std_initialization(config):
    settings = configuration(config)
    value = settings["sde_weight_std"] if settings["exploration"] == "gsde" else config["initial_action_std"]
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Initial noise scale must be finite and in (0, 1]")
    return math.log(value)


def require_matching(left, right):
    if configuration(left) != configuration(right):
        raise ValueError("Exploration configurations differ")


def validate_model(model, config):
    from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution
    settings = configuration(config)
    enabled = settings["exploration"] == "gsde"
    expected = StateDependentNoiseDistribution if enabled else DiagGaussianDistribution
    if bool(model.use_sde) != enabled or not isinstance(model.policy.action_dist, expected):
        raise ValueError("Saved exploration distribution differs from configuration")
    if enabled and model.sde_sample_freq != settings["sde_sample_freq"]:
        raise ValueError("Saved gSDE resampling frequency differs from configuration")
    return settings
