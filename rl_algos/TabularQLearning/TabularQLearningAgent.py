from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast
import random


class TabularQLearningAgent(object):
    def __init__(
        self,
        possible_actions,
        eps: float = 1.0,
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
            return random.choice(self._possible_actions(state))
        return self._best_action(state)


    def _best_action(self, state):
        try:
            possible_actions = self._possible_actions(state)
            best_action = possible_actions[0]
            best_q_value = self._q_values[state][best_action]
            for action in possible_actions:
                if self._q_values[state][action] > best_q_value:
                    best_action = action
                    best_q_value = self._q_values[state][action]
    
            return best_action
        except :
            print(state)
            print(self._possible_actions(state))
            raise Exception("")

    def get_value(self, state):
        return self._q_values[state][self._best_action(state)]


    def train(self, state, action, reward, next_state, done) -> None:
        target = reward + self._gamma * self.get_value(next_state) if not done else reward
        updated_q = (1 - self._lr) * self._q_values[state][action] + self._lr * target

        self._q_values[state][action] = updated_q

    def render_q_table(self):
        for state in self._q_values:
            res = f"{state}| "
            for action in sorted(self._q_values[state]):
                res += f"{self._q_values[state][action]}|"
            print(res)

 