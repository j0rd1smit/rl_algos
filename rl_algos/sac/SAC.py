import gym
import numpy as np
import tensorflow as tf
from tqdm import trange

from rl_algos.sac.SACAgent import SACAgent
from rl_algos.sac.SACConfig import SACConfig
from rl_algos.utils.ReplayBuffer import ReplayBuffer


class SAC(object):
    def __init__(
            self,
            env: gym.Env,
            test_env: gym.Env,
            agent: SACAgent,
            replay_buffer: ReplayBuffer,
            config: SACConfig,
            writer: tf.summary.SummaryWriter,
    ) -> None:
        self._env = env
        self._test_env = test_env
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._config = config
        self._writer = writer

        self._step = 0

    def run(
            self,
    ) -> None:
        v_loss_metric = tf.keras.metrics.Mean("v_loss")
        pi_loss_metric = tf.keras.metrics.Mean("pi_loss")
        o, r, d, ep_ret, ep_len = self._env.reset(), 0.0, False, 0.0, 0
        for t in trange(self._config.total_steps + self._config.start_steps):
            self._step += 1

            if t > self._config.start_steps:
                a = self.get_action(o, False)
            else:
                a = self._env.action_space.sample()

            o2, r, d, _ = self._env.step(a)
            r = self._config.reward_scaling_factor * r
            ep_ret += r
            ep_len += 1

            d = False if ep_len == self._config.max_ep_len else d

            self._replay_buffer.store(o, a, r, o2, d)

            o = o2
            if d or ep_len == self._config.max_ep_len:
                for j in range(ep_len):
                    states, actions, rewards, next_states, dones = self._replay_buffer.sample_batch(self._config.batch_size)
                    pi_loss, v_loss = self._agent.train(states, actions, rewards, next_states, dones)

                    pi_loss_metric(pi_loss)
                    v_loss_metric(v_loss)

                print(f"t={t} returns={ep_ret} ep_len={ep_len}")
                with self._writer.as_default():
                    tf.summary.scalar("reward_training", ep_ret, step=self._step)
                    tf.summary.scalar("eps_len_training", ep_len, step=self._step)
                    tf.summary.scalar("pi_loss", pi_loss_metric.result(), step=self._step)
                    tf.summary.scalar("v_loss", v_loss_metric.result(), step=self._step)

                o, r, d, ep_ret, ep_len = self._env.reset(), 0, False, 0, 0
                pi_loss_metric.reset_states()
                v_loss_metric.reset_states()
            if (t + 0) % 5000 == 0:
                self.run_test()

    def get_action(
            self,
            state: np.ndarray,
            determinstic: bool
    ) -> np.ndarray:
        return self._agent.select_actions(np.array([state]), determinstic)[0]

    def run_test(self) -> None:
        reward_metric = tf.keras.metrics.Mean("reward_test")
        eps_len_metrix = tf.keras.metrics.Mean("eps_len_test")
        render = True
        for i in range(self._config.n_test_episodes):
            o, r, d, ep_ret, ep_len = self._test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == self._config.max_ep_len)):
                if render:
                    self._test_env.render()
                # Take deterministic actions at test time
                a = self.get_action(o, True)
                o, r, d, _ = self._test_env.step(a)
                ep_ret += r
                ep_len += 1
            render = False
            reward_metric(ep_ret)
            eps_len_metrix(ep_len)

        with self._writer.as_default():
            tf.summary.scalar("reward_test", reward_metric.result(), step=self._step)
            tf.summary.scalar("eps_len_test", eps_len_metrix.result(), step=self._step)
