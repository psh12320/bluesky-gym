from gymnasium import spaces


class TrafficSlots:
    """Change observed traffic slots while retaining the original simulated traffic."""

    def __init__(self, *args, observation_traffic_slots, **kwargs):
        super().__init__(*args, **kwargs)
        if observation_traffic_slots < 1:
            raise ValueError("At least one observed traffic slot is required")
        self.intruder_obs.n = observation_traffic_slots
        traffic_spaces = self.intruder_obs.space()
        if hasattr(self, "observation_spaces"):
            self.observation_spaces = {agent: spaces.Dict({**space.spaces, **traffic_spaces})
                                       for agent, space in self.observation_spaces.items()}
        else:
            self.observation_space = spaces.Dict({**self.observation_space.spaces, **traffic_spaces})
