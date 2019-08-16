from typing import Tuple

import numpy as np

import rl_algos.utils.core as core


class ReplayBuffer(object):
    def __init__(
            self,
            obs_dim: Tuple[int],
            act_dim: Tuple[int],
            size: int,

    ) -> None:
        self.state_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.int32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.state_next_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.size = 0
        self.max_size = size
        self.ptr = 0

    def store(
            self,
            state: np.ndarray,
            action: float,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ) -> None:
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.state_next_buf[self.ptr] = next_state
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)

        return self.state_next_buf[idxs], self.action_buf[idxs], self.rewards_buf[idxs], self.state_next_buf[idxs], self.done_buf[idxs]
