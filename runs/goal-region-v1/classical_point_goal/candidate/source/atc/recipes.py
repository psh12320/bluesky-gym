from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Recipe:
    gamma: float = 0.99
    reach_reward: float = 1.0
    intrusion_penalty: float = -1.0
    restricted_area_penalty: float = -1.0
    sector_exit_penalty: float = -1.0
    drift_penalty: float = -0.01
    observation_traffic_slots: int | None = None
    conflict_features: bool = False
    goal_relative: bool = False
    route_guided: bool = False
    route_input: bool = False
    route_choice: bool = False
    route_residual: bool = False
    fast_speed_reference: bool = False
    route_sector_clearance: float = 0.0
    speed_increment_knots: float | None = None
    decision_interval_seconds: int | None = None

    def __post_init__(self):
        if self.decision_interval_seconds is not None and (
                type(self.decision_interval_seconds) is not int or self.decision_interval_seconds < 1):
            raise ValueError("Decision interval must be a positive integer number of seconds")
        if self.fast_speed_reference and not self.route_residual:
            raise ValueError("Fast speed reference requires residual control")
        if self.speed_increment_knots is not None and (
                not math.isfinite(self.speed_increment_knots) or self.speed_increment_knots <= 0):
            raise ValueError("Speed-command increment must be finite and positive")

    def action_kwargs(self):
        result = {} if self.speed_increment_knots is None else {"d_speed": self.speed_increment_knots}
        if self.decision_interval_seconds is not None:
            result["action_frequency"] = self.decision_interval_seconds
        return result

    def action_configuration(self):
        result = {} if self.speed_increment_knots is None else {
            "speed_command_increment_knots": self.speed_increment_knots}
        if self.decision_interval_seconds is not None:
            result["decision_interval_seconds"] = self.decision_interval_seconds
        return result

    def reward_kwargs(self):
        return {name: getattr(self, name) for name in (
            "reach_reward", "intrusion_penalty", "restricted_area_penalty",
            "sector_exit_penalty", "drift_penalty",
        )}


RECIPES = {
    "baseline": Recipe(),
    "balanced": Recipe(gamma=0.997, reach_reward=25.0),
    "goal_relative": Recipe(gamma=0.997, reach_reward=25.0, goal_relative=True),
    "route_guided": Recipe(gamma=0.997, reach_reward=25.0, route_guided=True),
    "route_guided_inset": Recipe(gamma=0.997, reach_reward=25.0, route_guided=True, route_sector_clearance=6.0),
    # Reward/discount reference from CGCooke/bluesky-gym, explib.CHAMPION_PARAMS,
    # commit a458870c711679e4fedb764abc1d58c8f4ffdb61 (see PUBLIC_WORK.md).
    # This uses our BlueSky collector, not the author's Rust training simulator.
    "public_weights": Recipe(
        gamma=0.996508469331006, reach_reward=37.41716380974046,
        intrusion_penalty=-0.9456977940944832,
        restricted_area_penalty=-0.5, sector_exit_penalty=-0.13428297844288217,
        drift_penalty=-0.0010342143754210312,
    ),
}

# Observation-only ablation: identical reward, discount and direct controls.
RECIPES["public_cpa"] = replace(RECIPES["public_weights"], conflict_features=True)
RECIPES["goal_relative_cpa"] = replace(RECIPES["goal_relative"], conflict_features=True)

# SA still simulates all ten scripted intruders; expose the nine closest to reuse MA weights.
RECIPES["public_sa_transfer"] = replace(RECIPES["public_weights"], observation_traffic_slots=9)
RECIPES["public_cpa_sa_transfer"] = replace(RECIPES["public_cpa"], observation_traffic_slots=9)

# Original direct-action policy layout, with a static-route bearing when needed.
RECIPES["public_route_input"] = replace(RECIPES["public_weights"], route_input=True)
RECIPES["public_route_input_sa_transfer"] = replace(RECIPES["public_route_input"], observation_traffic_slots=9)

# Learn adjustments around the route follower; zero action preserves its command.
RECIPES["public_route_residual"] = replace(RECIPES["public_route_input"], route_residual=True)
RECIPES["public_route_residual_sa_transfer"] = replace(RECIPES["public_route_residual"], observation_traffic_slots=9)

# Compare route detours across clearances without changing control or scoring.
RECIPES["public_route_choice"] = replace(RECIPES["public_route_input"], route_choice=True)
RECIPES["public_route_choice_sa_transfer"] = replace(RECIPES["public_route_choice"], observation_traffic_slots=9)
RECIPES["public_route_choice_residual"] = replace(RECIPES["public_route_residual"], route_choice=True)
RECIPES["public_route_choice_residual_sa_transfer"] = replace(RECIPES["public_route_choice_residual"], observation_traffic_slots=9)

# Larger speed-command range; the original A320 acceleration limits still apply.
RECIPES["public_route_choice_speed20"] = replace(RECIPES["public_route_choice"], speed_increment_knots=20.0)
RECIPES["public_route_choice_speed20_sa_transfer"] = replace(RECIPES["public_route_choice_speed20"], observation_traffic_slots=9)
RECIPES["public_route_choice_speed20_residual"] = replace(RECIPES["public_route_choice_residual"], speed_increment_knots=20.0)
RECIPES["public_route_choice_speed20_residual_sa_transfer"] = replace(RECIPES["public_route_choice_speed20_residual"], observation_traffic_slots=9)

# Initialize residual learning at the faster classical reference.
RECIPES["public_route_choice_fast_residual"] = replace(RECIPES["public_route_choice_residual"], fast_speed_reference=True)
RECIPES["public_route_choice_fast_residual_sa_transfer"] = replace(RECIPES["public_route_choice_fast_residual"], observation_traffic_slots=9)

# More frequent decisions retain the original one-second scoring resolution.
RECIPES["public_route_choice_interval5"] = replace(RECIPES["public_route_choice"], decision_interval_seconds=5)
RECIPES["public_route_choice_interval5_sa_transfer"] = replace(RECIPES["public_route_choice_interval5"], observation_traffic_slots=9)
RECIPES["public_route_choice_fast_residual_interval5"] = replace(RECIPES["public_route_choice_fast_residual"], decision_interval_seconds=5)
RECIPES["public_route_choice_fast_residual_interval5_sa_transfer"] = replace(RECIPES["public_route_choice_fast_residual_interval5"], observation_traffic_slots=9)

# Arrival-reward ablation; all navigation, safety and control settings are retained.
RECIPES["public_route_choice_fast_residual_reach250"] = replace(RECIPES["public_route_choice_fast_residual"], reach_reward=250.0)
RECIPES["public_route_choice_fast_residual_reach250_sa_transfer"] = replace(RECIPES["public_route_choice_fast_residual_reach250"], observation_traffic_slots=9)
