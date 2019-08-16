from gym.spaces import Box


class DDPGConfig(object):
    def __init__(
            self,
            observation_space: Box,
            action_space: Box,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

        self.gamma = 0.99

        self.noise_scale = 0.1
