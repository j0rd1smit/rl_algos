from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast

import gym
import random
import numpy as np



class TicTacToEnvSelfPlay(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TicTacToEnvSelfPlay, self).__init__()
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int32)

        self._state = np.zeros((3, 3), dtype=np.float32)
        self._player_one_turn = True
        self._steps = 0

    @property
    def player_one_turn(self):
        return self._player_one_turn

    def valid_actions(self, state):
        actions = []
        for i in range(len(state)):
            if state[i] == 0:
                actions.append(i)
        return actions

    @property
    def state(self):
        if self._player_one_turn:
            return self._state.copy().flatten()
        return -1 * self._state.copy().flatten()

    @property
    def state_p1(self):
        return self._state.copy().flatten()

    @property
    def state_p2(self):
        return -1 * self._state.copy().flatten()

    def valid_actions(self):
        return np.argwhere(self.state == 0.0).flatten()

    def _game_has_ended(self, winning_score):
        if np.trace(self._state) == winning_score:
            return True
        if np.trace(np.rot90(self._state)) == winning_score:
            return True

        for i in range(3):
            if np.sum(self._state[i, :]) == winning_score:
                return True
            if np.sum(self._state[:, i]) == winning_score:
                return True

        return False

    def _is_a_draw(self):
        for i in range(3):
            for j in range(3):
                if self._state[i, j] == 0:
                    return False
        return True

    def step(self, action):
        self._steps += 1
        action = int(action)
        x = action // 3
        y = action % 3

        if self._steps > 150:
            self._episode_ended = True
            return self.state, -10.0, True, {}

        if self._state[x][y] != 0:
            #raise Exception("Called invalid move")
            return self.state, -0.1, False, {}

        new_value = 1 if self._player_one_turn else -1
        self._state[x][y] = new_value

        winning_value = 3 if self._player_one_turn else -3
        if self._game_has_ended(winning_value):
            return self.state, 1.0, True, {}

        if self._is_a_draw():
            return self.state, 0.0, True, {}

        self._player_one_turn = not self._player_one_turn
        return self.state, 0.0, False, {}


    def reset(self):
        self._steps = 0
        self._player_one_turn = random.choice([True, False])
        self._state = np.zeros((3, 3), dtype=np.float32)


        return self.state

    def render(self, mode='human', close=False):
        game_state = "On going"

        if self._game_has_ended(3):
            game_state = "P1 won"
        elif self._game_has_ended(-3):
            game_state = "P2 won"
        elif self._is_a_draw():
            game_state = "draw"
        print()
        print("Board:")
        print(self._state)
        turn = "P1" if self._player_one_turn else "P2"
        print(f"Turn={turn} game_state={game_state}")
        print()

if __name__ == '__main__':
    import random
    env = TicTacToEnvSelfPlay()

    while len(env.valid_actions()) > 0:
        env.render()
        o,r, d, _ = env.step(random.choice(env.valid_actions()))
        if d:
            env.render()
            break