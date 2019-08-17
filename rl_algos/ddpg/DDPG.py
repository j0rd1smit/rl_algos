import gym
import numpy as np
import tensorflow as tf
from tqdm import trange

from rl_algos.ddpg.DDPGAgent import DDPGAgent
from rl_algos.ddpg.DDPGConfig import DDPGConfig
from rl_algos.utils.ReplayBuffer import ReplayBuffer


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

    def run(self) -> None:
        q_loss_metric = tf.keras.metrics.Mean("q_loss")
        pi_loss_metric = tf.keras.metrics.Mean("pi_loss")

        o, r, d, ep_ret, ep_len = self._env.reset(), 0.0, False, 0.0, 0

        for t in trange(self._config.steps + self._config.warmup_steps):
            if t > self._config.warmup_steps:
                a = self._agent.select_action(np.array([o]), noise_scale=self._config.act_noise)[0]
            else:
                a = self._env.action_space.sample()

            o2, r, d, _ = self._env.step(a)
            ep_ret += r
            ep_len += 1

            d = False if ep_len == self._config.max_ep_len else d

            r = r * self._config.reward_scaling_factor
            self._buffer.store(o, a, r, o2, d)
            o = o2

            if d or (ep_len == self._config.max_ep_len):
                for j in range(ep_len):
                    states, actions, rewards, next_states, dones = self._buffer.sample_batch(self._config.batch_size)

                    q_loss = self._agent.train_q(states, actions, rewards, next_states, dones)
                    q_loss_metric(q_loss)
                    step = t - ep_len + j
                    with self._writer.as_default():
                        tf.summary.scalar("q_loss_metric", q_loss, step=step)


                    pi_loss = self._agent.train_pi(states)
                    self._agent.update_target()
                    with self._writer.as_default():
                        tf.summary.scalar("pi_loss_metric", pi_loss, step=step)
                    pi_loss_metric(pi_loss)

                with self._writer.as_default():
                    # tf.summary.scalar("q_loss_metric", q_loss_metric.result(), step=t)
                    # tf.summary.scalar("pi_loss_metric", pi_loss_metric.result(), step=t)
                    tf.summary.scalar("episode_returns", ep_ret, step=t)
                    tf.summary.scalar("episode_len", ep_len, step=t)

                print(f"returns: {ep_ret}")
                print(f"episode_len: {ep_len}")

                o, r, d, ep_ret, ep_len = self._env.reset(), 0, False, 0, 0
                q_loss_metric.reset_states()
                pi_loss_metric.reset_states()