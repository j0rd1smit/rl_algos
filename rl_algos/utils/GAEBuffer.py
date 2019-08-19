from typing import Tuple, Union

import numpy as np

import rl_algos.utils.core as core


class GAEBuffer(object):
    def __init__(
            self,
            obs_dim: Tuple[int],
            act_dim: Tuple[int],
            gamma: float,
            lam: float,
            size: int,

    ) -> None:
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.int32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam
        self.max_size = size
        self.ptr = 0
        self.path_start_idx = 0

    def store(
            self,
            obs: np.ndarray,
            action: Union[int, np.ndarray],
            reward: Union[float, np.ndarray],
            value: Union[float, np.ndarray],
            logp: Union[float, np.ndarray],
    ) -> None:
        assert self.ptr < self.max_size,  "Buffer is out of room to store more"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr] = value
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(
            self,
            last_val: float = 0.0,
    ) -> None:
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rew_buf[path_slice], last_val)
        values = np.append(self.val_buf[path_slice], last_val)

        # delta(v,t) = r_t + gamma * V(s_t_+_1) - V(s_t)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam) #GAE-Kambda advantage

        self.ret_buf[path_slice] = core.discount_cumsum(rewards, self.gamma)[:-1] # rewards-to-go targets value func

        self.path_start_idx = self.ptr

    def get(
            self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.ptr == self.max_size, "Buffer is not yet full"
        self.ptr = 0
        self.path_start_idx = 0

        adv_mean, adv_std = core.statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf


