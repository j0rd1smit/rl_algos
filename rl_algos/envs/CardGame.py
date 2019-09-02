from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast

import gym
import numpy as np


class CardGame(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CardGame, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = gym.spaces.Discrete(2)
        # Example for using image as input:
        self.observation_space = gym.spaces.Box(low=0, high=32, shape=(1,), dtype=np.int32)

        self._state = 0
        self._episode_ended = False

    def step(self, action):
        if action == 1:
            self._episode_ended = True
        elif action == 0:
            new_card = np.random.randint(1, 11)
            self._state += new_card

        if self._episode_ended or self._state >= 21:
            reward = self._state - 21 if self._state <= 21 else -21
            return np.array([self._state]), reward, True, {}

        return np.array([self._state]), 0, False, {}

    def reset(self):
        self._state = 0
        self._episode_ended = False
        return np.array([self._state])

    def render(self, mode='human', close=False):
        print(f"state={self._state}")
 