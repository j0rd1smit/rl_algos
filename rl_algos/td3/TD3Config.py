from gym.spaces import Box


class TD3Config(object):
    def __init__(
            self,
            observation_space: Box,
            action_space: Box,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

        self.gamma = 0.99

        self.act_noise = (action_space.high - action_space.low) / 20.0

        self.target_noise_std = (action_space.high - action_space.low) / 20.0
        self.target_noise_clip = 0.5

        self.lr_pi = 1e-3
        self.lr_q = 1e-3

        self.polyak = 0.995

        self.batch_size = 64
        self.buffer_size = int(1e6)

        self.max_ep_len = 1000
        self.steps = 100000
        self.warmup_steps = 5000

        self.policy_deplay = 2

        self.reward_scaling_factor = 1.0
