from gym.spaces import Box


class DDPGConfig(object):
    def __init__(
            self,
            observation_space: Box,
            action_space: Box,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

        self.gamma = 0.999
        self.noise_scale = (action_space.high - action_space.low) / 20.0

        self.lr_pi = 1e-3
        self.lr_q = 1e-3

        self.warm_up_steps = 10000
        self.reward_scaling = 1.0

        self.max_len_episode = 1000

        self.batch_size = 64

        self.polyak = 0.995
