import gym
import numpy as np
from tqdm import trange

from rl_algos.DDPG.DDPGAgent import DDPGAgent
from rl_algos.DDPG.DDPGConfig import DDPGConfig


class DDPG(object):
    def __init__(
            self,
            env: gym.Env,
            agent: DDPGAgent,
            config: DDPGConfig,
    ) -> None:
        self._env = env
        self._agent = agent
        self._config = config

    def train(self, steps: int) -> None:
        s = self._env.reset()
        for i in trange(steps):
            self._env.render()
            a = self._agent.get_action(np.array([s]))
            print(a)
            next_s, r, d, _ = self._env.step(a)
            if d:
                s = self._env.reset()
