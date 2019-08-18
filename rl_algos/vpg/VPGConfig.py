from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast
from gym.spaces import Box


class VPGConfig(object):
    def __init__(
            self,
        observation_space: Box,
        action_space: Box,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

        self.lr_pi = 3e-4
        self.lr_v = 1e-3

        self.gamma = 0.99
        self.lamb= 0.99

        self.batch_size = 64
        self.train_v_iters = 80

        self.max_ep_len = 1000
        self.steps_per_epoch = 4000
        self.epochs = 50



        self.reward_scaling_factor = 1.0
 