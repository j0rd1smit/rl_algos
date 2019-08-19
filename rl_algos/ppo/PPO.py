import gym
import numpy as np
import tensorflow as tf
from tqdm import trange

from rl_algos.ppo.PPOAgent import PPOAgent
from rl_algos.ppo.PPOConfig import PPOConfig
from rl_algos.utils.GAEBuffer import GAEBuffer


class PPO(object):
    def __init__(
            self,
            env: gym.Env,
            agent: PPOAgent,
            buffer: GAEBuffer,
            config: PPOConfig,
            writer: tf.summary.SummaryWriter,
            render: bool = False
    ) -> None:
        self._env = env
        self._agent = agent
        self._buffer = buffer
        self._config = config
        self._writer = writer
        self._render = render

    def run(self) -> None:
        o, r, d, ep_ret, ep_len = self._env.reset(), 0.0, False, 0.0, 0
        reward_metric = tf.keras.metrics.Mean("reward")
        eps_len_metric = tf.keras.metrics.Mean("eps_len_metrix")

        for epoch in trange(self._config.epochs):
            for t in range(self._config.steps_per_epoch):
                # self._env.render()
                a, v_t, logp_t = self._agent.select_actions(np.array([o]))

                self._buffer.store(o, a, r, v_t, logp_t)

                o, r, d, _ = self._env.step(a[0])
                ep_ret += r * self._config.reward_scaling_factor
                ep_len += 1

                terminal = d or (ep_len == self._config.max_ep_len)
                if terminal or (t == self._config.steps_per_epoch - 1):

                    reward_metric(ep_ret)
                    eps_len_metric(ep_len)

                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = r if d else self._agent.predict_values(np.array([o]))[0]
                    self._buffer.finish_path(last_val)
                    o, r, d, ep_ret, ep_len = self._env.reset(), 0.0, False, 0.0, 0

            print("")
            print(f"avg episode reward: {reward_metric.result().numpy()}")
            print(f"avg episode length: {eps_len_metric.result().numpy()}")
            with self._writer.as_default():
                tf.summary.scalar("avg_episode_reward", reward_metric.result(), step=epoch)
                tf.summary.scalar("avg_episode_reward", eps_len_metric.result(), step=epoch)
            reward_metric.reset_states()
            eps_len_metric.reset_states()
            self.update()

            o, r, d, ep_ret, ep_len = self._env.reset(), 0.0, False, 0.0, 0
            while not (d or (ep_len == self._config.max_ep_len)):
                self._env.render()
                a, _, _ = self._agent.select_actions(np.array([o]))
                o, _, d, _ = self._env.step(a[0])
                ep_len += 1
            o, r, d, ep_ret, ep_len = self._env.reset(), 0.0, False, 0.0, 0

    def update(
            self
    ) -> None:
        states, actions, advantage, returns, logps = self._buffer.get()
        self._agent.train(states, actions, advantage, returns, logps)
