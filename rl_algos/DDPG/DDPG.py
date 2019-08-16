import gym
from tqdm import trange

from rl_algos.DDPG.DDPGAgent import DDPGAgent
from rl_algos.DDPG.DDPGConfig import DDPGConfig
from rl_algos.DDPG.ReplayBuffer import ReplayBuffer


class DDPG(object):
    def __init__(
            self,
            env: gym.Env,
            agent: DDPGAgent,
            buffer: ReplayBuffer,
            config: DDPGConfig,
    ) -> None:
        self._env = env
        self._agent = agent
        self._buffer = buffer
        self._config = config

        self._steps = 0

    def train(self, steps: int) -> None:
        s, episode_returns, episode_len = self._env.reset(), 0, 0

        print("warming up")
        if self._buffer.size < self._config.warm_up_steps:
            for _ in trange(self._buffer.size - self._steps):
                a = self._env.action_space.sample()
                next_s, r, d, _ = self._env.step(a)
                d = episode_len == self._config.max_len_episode or d
                self._buffer.store(s, a, r, next_s, d)
                s = next_s
                episode_returns += r
                episode_len += 1
                if d:
                    s = self._env.reset()
                    episode_len, episode_returns = 0, 0

        print("starting training")
        s, episode_returns, episode_len = self._env.reset(), 0, 0
        for i in trange(steps):
            a = self._env.action_space.sample()

            next_s, r, d, _ = self._env.step(a)
            episode_returns += r
            episode_len += 1

            d = episode_len == self._config.max_len_episode or d

            self._buffer.store(s, a, r, next_s, d)
            s = next_s

            if d:

                for _ in range(episode_len):
                    states, actions, rewards, next_states, dones = self._buffer.sample_batch(self._config.batch_size)
                    # TODO preforms training
                    q_loss = self._agent.train_q(states, actions, rewards, next_states, dones)
                    pi_loss = self._agent.train_pi(states)

                # TODO logging results
                print(f"returns: {episode_returns}")
                print(f"episode_len: {episode_len}")

                s, episode_returns, episode_len = self._env.reset(), 0, 0
