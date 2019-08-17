import gym
from tqdm import trange

from rl_algos.DDPG.DDPGAgent import DDPGAgent
from rl_algos.DDPG.DDPGConfig import DDPGConfig
from rl_algos.DDPG.ReplayBuffer import ReplayBuffer
import tensorflow as tf
import numpy as np

class DDPG(object):
    def __init__(
            self,
            env: gym.Env,
            agent: DDPGAgent,
            buffer: ReplayBuffer,
            config: DDPGConfig,
            writer: tf.summary.SummaryWriter,
    ) -> None:
        self._env = env
        self._agent = agent
        self._buffer = buffer
        self._config = config
        self._writer = writer

        self._steps = 0

    def train(self, steps: int) -> None:
        print("starting training")
        s, episode_returns, episode_len = self._env.reset(), 0, 0
        q_loss_metric = tf.keras.metrics.Mean("q_loss")
        pi_loss_metric = tf.keras.metrics.Mean("pi_loss")
        for i in trange(steps):
            a = self._agent.get_action(np.array([s]), noise_scale=self._config.noise_scale)[0]

            next_s, r, d, _ = self._env.step(a)
            r = r * self._config.reward_scaling
            episode_returns += r
            episode_len += 1

            d = episode_len == self._config.max_len_episode or d

            self._buffer.store(s, a, r, next_s, d)
            s = next_s
            

            if d:
                for _ in range(episode_len):
                    states, actions, rewards, next_states, dones = self._buffer.sample_batch(self._config.batch_size)
                    q_loss = self._agent.train_q(states, actions, rewards, next_states, dones)
                    pi_loss = self._agent.train_pi(states)
                    self._agent.update_target()
                    q_loss_metric(q_loss)
                    pi_loss_metric(pi_loss)


                # TODO logging results
                print(f"returns: {episode_returns}")
                print(f"episode_len: {episode_len}")
                self._steps += 1
                with self._writer.as_default():
                    tf.summary.scalar("q_loss_metric", q_loss_metric.result(), step=self._steps)
                    tf.summary.scalar("pi_loss_metric", pi_loss_metric.result(), step=self._steps)
                    tf.summary.scalar("episode_returns", episode_returns, step=self._steps)
                    tf.summary.scalar("episode_len", episode_len, step=self._steps)

                s, episode_returns, episode_len = self._env.reset(), 0, 0
                q_loss_metric.reset_states()
                pi_loss_metric.reset_states()
