from multiprocessing.dummy import Pool as ThreadPool
from typing import List, Tuple

import gym
import numpy as np


class BatchEnv(object):
    def __init__(
            self,
            envs: List[gym.Env],
            pool_size: int = 4
    ) -> None:
        self._workers = [_Worker(env) for env in envs]
        self._pool = ThreadPool(pool_size)

    def reset(self) -> np.ndarray:
        return np.array([worker.reset() for worker in self._workers]).astype(np.float32)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions_per_env = zip(self._workers, actions)
        stepped_envs = self._pool.map(lambda worker_actions: worker_actions[0].step(worker_actions[1]), actions_per_env)
        new_obs, rewards, done = (map(np.array, zip(*stepped_envs)))

        return new_obs.astype(np.float32), rewards.astype(np.float32), done

    def allow_reset_after_step(self) -> None:
        for worker in self._workers:
            worker.allow_reset_after_step()


class _Worker(object):
    def __init__(
            self,
            env: gym.Env,
    ) -> None:
        self._env = env
        self._is_done = False
        self._next_starting_state = np.zeros_like(self._env.observation_space.shape)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        if not self._is_done:
            s, r, done, _ = self._env.step(action[0])
            self._is_done = done
            if done:
                self._next_starting_state = self._env.reset()
        else:
            s = self._next_starting_state
            r = 0
            done = True

        return s, r, done

    def reset(self) -> np.ndarray:
        s = self._env.reset()
        self._is_done = False

        return s

    def allow_reset_after_step(self) -> None:
        if self._is_done:
            self._is_done = False
