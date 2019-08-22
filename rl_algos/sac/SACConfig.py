from gym.spaces import Box


class SACConfig(object):
    def __init__(
            self,
            observation_space: Box,
            action_space: Box,

    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

        self.lr_pi = 1e-3
        self.lr_v = 1e-3

        self.n_test_episodes = 5

        self.max_ep_len = 1000

        self.total_steps = 100 * 5000

        self.start_steps = 10_000

        self.reward_scaling_factor = 1.0

        self.buffer_size = 1_000_000
        self.batch_size = 100

        self.entropy_regularization_weight = 0.2

        self.gamma = 0.99
        self.polyak = 0.995
