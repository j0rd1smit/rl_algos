from multiprocessing.dummy import Pool as ThreadPool
from typing import Dict, List, Tuple

import gym
import numpy as np


def _step_single_env(data: Tuple[gym.Env, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
    env, action = data
    s, r, done, info = env.step(action[0])
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
        self._pool = ThreadPool(pool_size)

    def reset(self) -> np.ndarray:
        return np.array([env.reset() for env in self._envs]).astype(np.float32)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        actions_per_env = zip(self._envs, actions)
        stepped_envs = self._pool.map(_step_single_env, actions_per_env)
        new_obs, rewards, done, info = (map(np.array, zip(*stepped_envs)))

        return new_obs.astype(np.float32), rewards.astype(np.float32), done, list(info)




if __name__ == '__main__':
    n_envs = 64
    envs = [gym.make("CartPole-v1") for _ in range(n_envs)]
    batch_env = BatchEnv(envs)
    _ = batch_env.reset()

    for _ in range(100):
        actions = np.array([[env.action_space.sample()] for env in envs])
        _, _, _, _ = batch_env.step(actions)
