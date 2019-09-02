import gym
import numpy as np
import random


class TicTacToEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TicTacToEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = gym.spaces.Discrete(9)
        # Example for using image as input:
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int32)

        self._state = np.zeros((3, 3), dtype=np.float32)
        self._episode_ended = False
        self._steps = 0

    @property
    def state(self):
        return self._state.copy().flatten()

    def _preform_random_move_oppenend(self):
        while True:
            x, y = np.random.randint(3, size=2)
            if self._state[x, y] == 0:
                self._state[x, y] = -1
                break

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

        if self._steps > 20:
            self._episode_ended = True
            return self.state, -10.0, True, {}

        if self._state[x][y] != 0:
            return self.state, -0.1, False, {}

        new_value = 1
        self._state[x][y] = new_value

        if self._game_has_ended(3):
            return self.state, 1.0, True, {}

        if self._is_a_draw():
            self._episode_ended = True
            return self.state, 0.0, True, {}

        self._preform_random_move_oppenend()

        if self._game_has_ended(-3):
            return self.state, -1.0, True, {}

        if self._is_a_draw():
            self._episode_ended = True
            return self.state, 0.0, True, {}

        return self.state, 0.0, False, {}


    def reset(self):
        self._steps = 0
        self._state = np.zeros((3, 3), dtype=np.float32)
        self._episode_ended = False

        has_first_turn = random.choice([True, False])
        if has_first_turn:
            self._preform_random_move_oppenend()

        return self.state

    def render(self, mode='human', close=False):
        print()
        print(f"step={self._steps} ended: {self._episode_ended} won: {self._game_has_ended(3)} lost: {self._game_has_ended(-3)} draw: {self._is_a_draw() and not self._game_has_ended(3) and not self._game_has_ended(-3)} ")
        print(self._state)
        print()

