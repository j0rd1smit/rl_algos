import multiprocessing
from typing import Dict, List, Tuple

import gym
import numpy as np


def _step_single_env(data: Tuple[gym.Env, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
    env, action = data
    s, r, done, info = env.step(action)
    if done:
        s = env.reset()

    return s, r, done, info


class BatchEnv(object):
    def __init__(
            self,
            envs: List[gym.Env],
            pool_size: int = 8
    ) -> None:
        self._envs = envs
        self._pool = multiprocessing.Pool(pool_size)

    def reset(self) -> np.ndarray:
        return np.array([env.reset() for env in self._envs])

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        actions_per_env = zip(self._envs, actions)
        stepped_envs = self._pool.map(_step_single_env, actions_per_env)
        new_obs, rewards, done, info = (map(np.array, zip(*stepped_envs)))

        return new_obs, rewards, done, list(info)

    def _step_single_env(self, env: gym.Env, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        s, r, done, info = env.step(action)
        if done:
            s = env.reset()

        return s, r, done, info


if __name__ == '__main__':
    n_envs = 16
    envs = [gym.make("CartPole-v1") for _ in range(n_envs)]
    batch_env = BatchEnv(envs)
    s = batch_env.reset()

    actions = np.ones((n_envs,), dtype=np.int32)
    new_obs, rewards, done, info = batch_env.step(actions)
    print(new_obs)
    print(rewards)
    print(done)
    print(info)
