from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast
import operator as op
from functools import reduce

from rl_algos.cross_entropy_table.CrossEntropyTable import CrossEntropyTable, Episode


def _ncr(n: int, r: int) -> int:
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom

def _n_states(n_houses: int, n_moves: int) -> int:
    return sum([_ncr(n_houses, i) for i in range(0, n_moves + 1)])


class StateMapping(object):
    def __init__(
        self,
        max_n_states: int,
        init_states: Optional[List[Tuple]] = None,
    ) -> None:
        self._max_n_states = max_n_states
        self._state_counter = 0
        self._tuple_state_mapping: Dict[tuple, int] = {}
        self._state_tuple_mapping: Dict[int, tuple] = {}

        if init_states is not None:
            for s in init_states:
                self.add_state(s)

    def decode_state(self, s: int) -> tuple:
        assert s in self._state_tuple_mapping
        return self._state_tuple_mapping[s]

    def encode_state(self, s: tuple) -> int:
        assert s in self._tuple_state_mapping
        return self._tuple_state_mapping[s]

    def has_state(self, s: tuple) -> bool:
        return s in self._tuple_state_mapping

    def add_state(self, s: tuple) -> None:
        assert not self.has_state(s)
        assert self._state_counter < self._max_n_states
        encoded_s = self._state_counter
        self._tuple_state_mapping[s] = encoded_s
        self._state_tuple_mapping[encoded_s] = s

        self._state_counter += 1



class ThieveryEnv(object):
    def __init__(
            self,
            house_values: List[float],
            max_n_guards: int,
            max_n_robberies: int,

    ) -> None:
        self._house_values = house_values
        self._n_houses = len(house_values)

        self._max_n_guards = max_n_guards
        self._max_n_robberies = max_n_robberies


        self.n_states_guard = _n_states(self._n_houses, max_n_guards)
        self.n_actions_guard = len(house_values)
        self._state_mapping_guard = StateMapping(self.n_states_guard, init_states=[self.init_state()])



        self.n_states_thief = _n_states(self._n_houses, max_n_robberies)
        self.n_actions_thief = len(house_values)
        self._state_mapping_thief = StateMapping(self.n_states_thief, init_states=[self.init_state()])


    def init_state(self) -> tuple:
        return ()


    def generate_episode(self, guard: CrossEntropyTable, thief: CrossEntropyTable) -> Tuple[Episode, Episode]:
        states_guard, actions_guard, rewards_guard = [], [], []
        s_guard = self._state_mapping_guard.encode_state(self.init_state())

        for _ in range(self._max_n_guards):
            a = guard.get_action(s_guard)
            s_next = self._state_transition(s_guard, a, self._state_mapping_guard)

            states_guard.append(s_guard)
            actions_guard.append(a)
            rewards_guard.append(0.0)
            s_guard = s_next

        states_thief, actions_thief, rewards_thief = [], [], []
        s_thief = self._state_mapping_thief.encode_state(self.init_state())
        total_reward_thief = 0.0

        for _ in range(self._max_n_robberies):
            a = thief.get_action(s_guard)

            states_thief.append(s_thief)
            actions_thief.append(a)
            rewards_thief.append(0.0)

            if a in actions_guard:
                total_reward_thief = 0.0
                break
            else:
                total_reward_thief += self._house_values[a]

            s_thief = self._state_transition(s_thief, a, self._state_mapping_thief)

        rewards_guard[-1] = -1 * total_reward_thief
        rewards_thief[-1] = total_reward_thief

        return Episode(states_guard, actions_guard, rewards_guard), Episode(states_thief, actions_thief, rewards_thief)


    @staticmethod
    def _state_transition(s: int, action: int, state_mapping: StateMapping) -> int:
        state = state_mapping.decode_state(s)
        if action in state:
            state_next = state
        else:
            state_next = state + (action,)

        state_next = tuple(sorted(state_next))
        if not state_mapping.has_state(state_next):
            state_mapping.add_state(state_next)

        return state_mapping.encode_state(state_next)


