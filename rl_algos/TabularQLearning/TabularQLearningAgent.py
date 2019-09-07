from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast
import random


class TabularQLearningAgent(object):
    def __init__(
        self,
        possible_actions,
        eps: float = 0.25,
        lr: float = 0.5,
        gamma: float = 0.99,
    ) -> None:
        self._possible_actions = possible_actions
        self._eps = eps
        self._lr = lr
        self._gamma = gamma

        self._q_values = defaultdict(lambda: defaultdict(lambda: 0))


    def select_action(self, state) -> int:
        if random.uniform(0, 1) < self._eps:
            self._eps = self._eps * 0.995
            return random.choice(self._possible_actions)
        return self._best_action(state)


    def _best_action(self, state):
        best_action = self._possible_actions[0]
        best_q_value = self._q_values[state][best_action]
        for action in self._possible_actions:
            if self._q_values[state][action] > best_q_value:
                best_action = action
                best_q_value = self._q_values[state][action]

        return best_action

    def get_value(self, state):
        return self._q_values[state][self._best_action(state)]


    def train(self, state, action, reward, next_state) -> None:
        target = reward + self._gamma * self.get_value(next_state)
        updated_q = (1 - self._lr) * self._q_values[state][action] + self._lr * target

        self._q_values[state][action] = updated_q
 