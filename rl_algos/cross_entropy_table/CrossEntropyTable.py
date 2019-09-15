from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast
import numpy as np
from collections import defaultdict


class Episode(object):

    def __init__(
            self,
            states: List[int],
            actions: List[int],
            rewards: List[float],
    ) -> None:
        assert len(states) == len(actions) == len(rewards)
        self.states = states
        self.actions = actions
        self.rewards = rewards

    @property
    def total_reward(
            self,
        ) -> float:
        return sum(self.rewards)

    def __repr__(self) -> str:
        return f"Episode(states={self.states}, actions={self.actions}, rewards={self.rewards})"


class CrossEntropyTable(object):
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float,
        percentile: int,
    ) -> None:
        self._n_states = n_states
        self._n_actions = n_actions
        self._policy_logits = np.ones((self._n_states, self._n_actions)) / self._n_actions
        self._lr = lr
        self._percentile = percentile


    def get_action(self, state: int) -> int:
        return np.random.choice(self._n_actions, 1, p=self._policy_logits[state])[0]


    def update(self, episodes: List[Episode]) -> None:
        elite_episodes = self.select_elites(episodes)

        new_policy_logits = np.zeros_like(self._policy_logits)
        n_rows, n_cols = new_policy_logits.shape
        state_action_dict: Dict[int, List[float]] = defaultdict(lambda: [0.0] * self._n_actions)

        for i, episode in enumerate(elite_episodes):
            for s, a in zip(episode.states, episode.actions):
                state_action_dict[s][a] += 1

        for i in range(n_rows):
            if i not in state_action_dict:
                new_policy_logits[i] = [1.0 / n_cols] * n_cols
            else:
                new_policy_logits[i] = [p / sum(state_action_dict[i]) for p in state_action_dict[i]]

        self._policy_logits = self._lr * new_policy_logits + (1 - self._lr) * self._policy_logits

    def select_elites(self, episodes: List[Episode]) -> List[Episode]:
        rewards = list(map(lambda x: x.total_reward, episodes))
        reward_threshold = sum(rewards) / len(rewards)#np.percentile(rewards, self._percentile)
        elite_episodes = list(filter(lambda x: x.total_reward >= reward_threshold, episodes))

        return elite_episodes

