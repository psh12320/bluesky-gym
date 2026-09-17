"""Explicit continuous or categorical exploration and checkpoint validation."""
import math


def configuration(config):
    mode = config.get("exploration", "gaussian")
    weight = config.get("sde_weight_std")
    frequency = config.get("sde_sample_freq")
    if mode == "gaussian":
        if weight is not None or frequency is not None:
            raise ValueError("gSDE settings require exploration=gsde")
    elif mode == "categorical":
        if weight is not None or frequency is not None:
            raise ValueError("Categorical decisions do not use gSDE settings")
        if config.get("action_reference", "direct") != "direct" or config.get("neutral_action_mean", False):
            raise ValueError("Categorical maneuvers require direct commands and their recorded route-choice prior")
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
    if settings["exploration"] == "categorical":
        return 0.0  # AircraftPolicy ignores log_std_init for a categorical head.
    value = settings["sde_weight_std"] if settings["exploration"] == "gsde" else config["initial_action_std"]
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Initial noise scale must be finite and in (0, 1]")
    return math.log(value)


def require_matching(left, right, *, allow_unbuilt=False):
    if configuration(left) != configuration(right):
        raise ValueError("Exploration configurations differ")
    if left.get("exploration") == "categorical":
        from atc_rl.maneuvers import ManeuverMapping
        expected = ManeuverMapping.from_specification(left["maneuver_mapping"])
        if allow_unbuilt and "maneuver_mapping" not in right:
            return
        if "maneuver_mapping" not in right or expected != ManeuverMapping.from_specification(right["maneuver_mapping"]):
            raise ValueError("Maneuver mappings differ")


def validate_model(model, config):
    from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution, MultiCategoricalDistribution
    settings = configuration(config)
    enabled = settings["exploration"] == "gsde"
    if settings["exploration"] == "categorical":
        from atc_rl.maneuvers import ManeuverMapping
        mapping = ManeuverMapping.from_specification(config["maneuver_mapping"])
        if (model.use_sde or not isinstance(model.policy.action_dist, MultiCategoricalDistribution)
                or model.action_space != mapping.action_space
                or model.observation_space["actor"].shape != (mapping.actor_dim,)):
            raise ValueError("Saved categorical distribution or observation layout differs")
        return {**settings, "maneuver_mapping": mapping.specification()}
    expected = StateDependentNoiseDistribution if enabled else DiagGaussianDistribution
    if bool(model.use_sde) != enabled or not isinstance(model.policy.action_dist, expected):
        raise ValueError("Saved exploration distribution differs from configuration")
    if enabled and model.sde_sample_freq != settings["sde_sample_freq"]:
        raise ValueError("Saved gSDE resampling frequency differs from configuration")
    return settings
