from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast

import gym
import tensorflow as tf
import numpy as np
from tqdm import trange

from rl_algos.envs.TicTacToEnvSelfPlay import TicTacToEnvSelfPlay
from rl_algos.ppo.PPOAgent import PPOAgent
from rl_algos.ppo.PPOConfig import PPOConfig
from rl_algos.utils.GAEBuffer import GAEBuffer


class PPOSelfPlay(object):
    def __init__(
            self,
            env: TicTacToEnvSelfPlay,
            agent: PPOAgent,
            buffer_p1: GAEBuffer,
            buffer_p2: GAEBuffer,
            config: PPOConfig,

    ) -> None:
        self._env = env
        self._agent = agent
        self._buffer_p1 = buffer_p1
        self._buffer_p2 = buffer_p2
        self._config = config

    def run(self) -> None:
        o, r, d, ep_ret_p1, ep_ret_p2, ep_len_p1, ep_len_p2 = self._env.reset(), 0.0, False, 0.0, 0.0, 0, 0
        reward_metric_p1 = tf.keras.metrics.Mean("reward_p1")
        reward_metric_p2 = tf.keras.metrics.Mean("reward_p2")
        eps_len_metric_p1 = tf.keras.metrics.Mean("eps_len_p1")
        eps_len_metric_p2 = tf.keras.metrics.Mean("eps_len_p2")

        for epoch in trange(self._config.epochs):
            turn = self._env.player_one_turn
            for t in range(self._config.steps_per_epoch):
                a, v_t, logp_t = self._agent.select_actions(np.array([o]))

                if turn:
                    self._buffer_p1.store(o, a, r, v_t, logp_t)
                else:
                    self._buffer_p2.store(o, a, r, v_t, logp_t)

                o, r, d, _ = self._env.step(a[0])

                if turn:
                    ep_ret_p1 += r * self._config.reward_scaling_factor
                    ep_len_p1 += 1
                else:
                    ep_ret_p2 += r * self._config.reward_scaling_factor
                    ep_len_p2 += 1



                terminal = d or (ep_len_p1 + ep_len_p2 == self._config.max_ep_len)
                if terminal or (t == self._config.steps_per_epoch - 1):

                    reward_metric_p1(ep_ret_p1)
                    reward_metric_p2(ep_ret_p2)
                    eps_len_metric_p1(ep_len_p1)
                    eps_len_metric_p2(ep_len_p2)

                    last_val = r if d else self._agent.predict_values(np.array([o]))[0]
                    if turn:
                        self._buffer_p1.finish_path(last_val)
                        self._buffer_p2.finish_path(0)
                    else:
                        self._buffer_p1.finish_path(0)
                        self._buffer_p2.finish_path(last_val)
                    o, r, d, ep_ret_p1, ep_ret_p2, ep_len_p1, ep_len_p2 = self._env.reset(), 0.0, False, 0.0, 0.0, 0, 0
                turn = self._env.player_one_turn
            print("")
            print(f"avg episode reward p1: {reward_metric_p1.result().numpy()}")
            print(f"avg episode reward p2: {reward_metric_p2.result().numpy()}")
            print(f"avg episode length p1: {eps_len_metric_p1.result().numpy()}")
            print(f"avg episode length p2: {eps_len_metric_p2.result().numpy()}")
            reward_metric_p1.reset_states()
            eps_len_metric_p1.reset_states()
            reward_metric_p2.reset_states()
            eps_len_metric_p2.reset_states()
            self.update()

            """
            o, r, d, ep_ret_p1, ep_ret_p2, ep_len_p1, ep_len_p2 = self._env.reset(), 0.0, False, 0.0, 0.0, 0, 0
            while not (d):
                self._env.render()
                a, _, _ = self._agent.select_actions(np.array([o]))
                o, _, d, _ = self._env.step(a[0])
            o, r, d, ep_ret_p1, ep_ret_p2, ep_len_p1, ep_len_p2 = self._env.reset(), 0.0, False, 0.0, 0.0, 0, 0
            """

    def update(
            self
    ) -> None:
        states_p1, actions_p1, advantage_p1, returns_p1, logps_p1 = self._buffer_p1.get()
        states_p2, actions_p2, advantage_p2, returns_p2, logps_p2 = self._buffer_p2.get()
        states = np.concatenate([states_p1, states_p2])
        actions = np.concatenate([actions_p1, actions_p2])
        advantage = np.concatenate([advantage_p1, advantage_p2])
        returns = np.concatenate([returns_p1, returns_p2])
        logps = np.concatenate([logps_p1, logps_p2])


        self._agent.train(states, actions, advantage, returns, logps)

