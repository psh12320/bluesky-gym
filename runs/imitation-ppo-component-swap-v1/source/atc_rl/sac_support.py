"""Fresh SAC components sharing the on-policy simulator and finite horizon."""
import math
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
from atc.algorithms import NavigationSAC, initialize_navigation_actor
from atc.replay import AircraftReplayBuffer


class LocalAircraftInputs(VecEnvWrapper):
    """Remove joint critic inputs, including the observations saved at terminals."""
    def __init__(self, environment):
        layout = environment.observation_space
        if not isinstance(layout, spaces.Dict) or not isinstance(layout["actor"], spaces.Box):
            raise ValueError("Expected WorldPool actor/critic observations")
        super().__init__(environment, observation_space=layout["actor"])

    def reset(self):
        observation = self.venv.reset()
        self.reset_infos = self.venv.reset_infos
        return observation["actor"]

    def step_wait(self):
        observation, rewards, dones, originals = self.venv.step_wait()
        infos = []
        for original in originals:
            info = dict(original)
            if "terminal_observation" in info:
                info["terminal_observation"] = info["terminal_observation"]["actor"].copy()
            infos.append(info)
        self.reset_infos = self.venv.reset_infos
        return observation["actor"], rewards, dones, infos


class FiniteAircraftReplay(AircraftReplayBuffer):
    """Exclude padding and stop value bootstrapping at the actual task deadline."""
    def __init__(self, *args, handle_timeout_termination=False, **kwargs):
        if handle_timeout_termination:
            raise ValueError("The 3000-second task deadline must remain terminal")
        super().__init__(*args, handle_timeout_termination=False, **kwargs)

    def add(self, obs, next_obs, action, reward, done, infos):
        if len(infos) != self.input_envs:
            raise ValueError("Wrong aircraft batch size")
        for index, info in enumerate(infos):
            if "inactive" not in info:
                raise ValueError("Missing aircraft activity flag")
            if info["inactive"]:
                continue
            if "aircraft_terminated" not in info or "aircraft_truncated" not in info:
                raise ValueError("Missing individual aircraft terminal flags")
            expected = bool(info["aircraft_terminated"] or info["aircraft_truncated"])
            if bool(done[index]) != expected:
                raise ValueError("Aircraft terminal flags disagree with vector done")
        super().add(obs, next_obs, action, reward, done, infos)


class SACAccounting(BaseCallback):
    """Count every collected transition; finish each collection before stopping."""
    def __init__(self):
        super().__init__()
        self.live_transitions = 0
        self.padded_transitions = 0
        self.world_decisions = 0
        self.completed_worlds = 0
        self.completed_aircraft = []
        self.completed_learning_returns = []

    def _on_step(self):
        infos = self.locals["infos"]
        if any("inactive" not in info for info in infos):
            raise RuntimeError("Missing aircraft activity flag")
        self.world_decisions += len(infos) // 10
        self.completed_worlds += sum(bool(info.get("world_completed", False)) for info in infos)
        for info in infos:
            if info["inactive"]:
                self.padded_transitions += 1
            else:
                self.live_transitions += 1
                if info.get("aircraft_done"):
                    self.completed_aircraft.append(info["metrics"])
                    self.completed_learning_returns.append(float(info["learning_episode_return"]))
        # Returning False here would discard the step before replay storage.
        return True


def build_sac(environment, *, seed, device="cpu", buffer_size=200000,
              learning_starts=5000, batch_size=256, gradient_steps=1,
              initial_action_std=.05, entropy_initial=.01):
    for value in (initial_action_std, entropy_initial):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Noise and entropy scales must be finite and positive")
    if initial_action_std > 1:
        raise ValueError("Initial latent action standard deviation must be <= 1")
    if not isinstance(environment.observation_space, spaces.Box):
        raise ValueError("SAC must receive local inputs only")
    model = NavigationSAC(
        "MlpPolicy", environment, seed=seed, device=device,
        learning_rate=3e-4, gamma=.996508469331006, tau=.005,
        buffer_size=buffer_size, learning_starts=learning_starts,
        batch_size=batch_size, train_freq=(1, "step"), gradient_steps=gradient_steps,
        ent_coef=f"auto_{entropy_initial}", target_entropy="auto",
        replay_buffer_class=FiniteAircraftReplay,
        policy_kwargs={"net_arch": {"pi": [128, 128], "qf": [256, 256]},
                       "activation_fn": torch.nn.Tanh, "n_critics": 2},
        navigation_warmup_std=initial_action_std,
        verbose=0,
    )
    initialize_navigation_actor(model)
    with torch.no_grad():
        model.actor.log_std.bias.fill_(math.log(initial_action_std))
    return model


def verify_accounting(model, accounting):
    replay = model.replay_buffer
    if not isinstance(replay, FiniteAircraftReplay):
        raise ValueError("Wrong replay semantics")
    expected = (accounting.live_transitions, accounting.padded_transitions)
    actual = (replay.live_transitions, replay.skipped_transitions)
    if actual != expected or model.num_timesteps != sum(expected):
        raise ValueError("Collected, stored and padded transition counts disagree")
    if replay.handle_timeout_termination:
        raise ValueError("Task deadlines would bootstrap")
    return {"live_transitions": actual[0], "padded_transitions": actual[1],
            "counted_transitions": model.num_timesteps, "replay_size": replay.size()}


def load_sac_for_inference(path, config, device="cpu"):
    """Load fresh-comparison checkpoints without allocating a training replay."""
    from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
    if config.get("algorithm") != "sac":
        raise ValueError("Expected a fresh SAC comparison checkpoint")
    model = NavigationSAC.load(path, device=device, buffer_size=1)
    if model.replay_buffer_class is not FiniteAircraftReplay or model.replay_buffer.handle_timeout_termination:
        raise ValueError("SAC checkpoint does not use the finite task horizon")
    if model.use_sde or not isinstance(model.actor.action_dist, SquashedDiagGaussianDistribution):
        raise ValueError("SAC action distribution differs from the recorded recipe")
    if not isinstance(model.observation_space, spaces.Box):
        raise ValueError("SAC inference requires local inputs only")
    if model.gamma != config["gamma"] or model.navigation_warmup_std != config["initial_action_std"]:
        raise ValueError("SAC checkpoint configuration differs")
    if model.policy.net_arch != {"pi": config["actor_widths"], "qf": config["critic_widths"]}:
        raise ValueError("SAC actor/critic architecture differs")
    if len(model.critic.q_networks) != config["number_of_critics"]:
        raise ValueError("SAC critic count differs")
    return model
