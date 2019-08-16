import gym
import numpy as np
import tensorflow as tf
from tqdm import trange

from rl_algos.ppo.PPOAgent import PPOAgent
from rl_algos.utils.GAEBuffer import GAEBuffer


class PPO(object):
    def __init__(
            self,
            config: "PPOConfig",
            agent: PPOAgent,
            buffer: GAEBuffer,
            env: gym.Env,
            writer: tf.summary.SummaryWriter,
            render: bool = False
    ) -> None:
        self._config = config
        self._agent = agent
        self._buffer = buffer
        self._env = env
        self._writer = writer
        self._render = render

    def train(self, epochs: int) -> None:
        o, r, d, ep_ret, ep_len = self._env.reset(), 0, False, 0, 0

        reward_metric = tf.keras.metrics.Mean("reward")
        eps_len_metric = tf.keras.metrics.Mean("eps_len_metrix")


        for epoch in trange(epochs):
            for t in range(self._config.steps_per_epoch):
                a, v_t, logp_t = self._agent.select_actions(np.array([o]))
                self._buffer.store(o, a[0], r, v_t[0], logp_t[0])

                o, r, d, _ = self._env.step(a[0][0])
                ep_ret += r
                ep_len += 1

                is_terminal = d or (ep_len >= self._config.max_ep_len)
                is_last_step = t == self._config.steps_per_epoch -1

                if is_terminal:
                    reward_metric(ep_ret)
                    eps_len_metric(ep_len)

                if is_terminal or is_last_step:
                    last_val = r if d else self._agent.predict_value(np.array([o]))[0]
                    self._buffer.finish_path(last_val)

                    o, r, d, ep_ret, ep_len = self._env.reset(), 0, False, 0, 0
            print("")
            print(f"avg episode reward: {reward_metric.result().numpy()}")
            print(f"avg episode length: {eps_len_metric.result().numpy()}")
            with self._writer.as_default():
                tf.summary.scalar("avg_episode_reward", reward_metric.result(), step=epoch)
                tf.summary.scalar("avg_episode_reward", eps_len_metric.result(), step=epoch)
            
            
            
            reward_metric.reset_states()
            eps_len_metric.reset_states()
            obs, actions, advantage, returns, logp_old = self._buffer.get()
            self._agent.train(obs, actions, advantage, returns, logp_old)

            if self._render:
                while not d:
                    self._env.render()
                    a, _, _ = self._agent.select_actions(np.array([o]))
                    o, r, d, _ = self._env.step(a[0][0])
                o, r, d, ep_ret, ep_len = self._env.reset(), 0, False, 0, 0


class PPOConfig(object):
    def __init__(
        self
    ) -> None:
        self.steps_per_epoch = 4000
        self.max_ep_len = 1000