from typing import cast

import numpy as np
import tensorflow as tf

import rl_algos.utils.Types as Types
from rl_algos.td3.TD3Config import TD3Config
from rl_algos.td3.TD3Model import TD3Model


class TD3Agent(object):
    def __init__(
            self,
            model: TD3Model,
            target: TD3Model,
            config: TD3Config,
    ) -> None:
        self._model = model
        self._target = target
        self._config = config

        self.optimizer_q = tf.keras.optimizers.Adam(lr=self._config.lr_q)
        self.optimizer_pi = tf.keras.optimizers.Adam(lr=self._config.lr_pi)

    @cast(Types.Function, tf.function)
    def train_pi(
            self,
            states: Types.Tensor,
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self._pi_loss(states)

        variables = self._model.pi_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer_pi.apply_gradients(zip(gradients, variables))

        return loss

    def _pi_loss(
            self,
            states: tf.Tensor,
    ) -> tf.Tensor:
        pi = self._model.pi(states)
        q_values = self._model.q1(states, pi)  # Why isn't this not also a min?
        return - tf.reduce_mean(q_values)

    @cast(Types.Function, tf.function)
    def train_q(
            self,
            states: Types.Tensor,
            actions: Types.Tensor,
            rewards: Types.Tensor,
            next_states: Types.Tensor,
            dones: Types.Tensor,
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self._q_loss(states, actions, rewards, next_states, dones)

        variables = self._model.q1_variables + self._model.q2_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer_q.apply_gradients(zip(gradients, variables))

        return loss

    def _q_loss(
            self,
            states: tf.Tensor,
            actions: tf.Tensor,
            rewards: tf.Tensor,
            next_states: tf.Tensor,
            dones: tf.Tensor,
    ) -> tf.Tensor:
        pi_target = self._target.pi(next_states)
        epsilon = tf.random.normal(pi_target.shape, stddev=self._config.target_noise_std)
        epsilon = tf.clip_by_value(epsilon, - self._config.target_noise_clip, self._config.target_noise_clip)
        pi_target = pi_target + epsilon
        pi_target = tf.clip_by_value(pi_target, self._config.action_space.low, self._config.action_space.high)

        q1_target = self._target.q1(next_states, pi_target)
        q2_target = self._target.q2(next_states, pi_target)
        min_q_target = tf.stop_gradient(tf.minimum(q1_target, q2_target))
        # noinspection PyTypeChecker
        bellman_backup = rewards + self._config.gamma * (1.0 - dones) * min_q_target

        q1_values = self._model.q1(states, actions)
        q2_values = self._model.q2(states, actions)

        # noinspection PyTypeChecker
        q1_loss = tf.reduce_mean((q1_values - bellman_backup) ** 2)
        # noinspection PyTypeChecker
        q2_loss = tf.reduce_mean((q2_values - bellman_backup) ** 2)

        return q1_loss + q2_loss

    @cast(Types.Function, tf.function)
    def update_target(self) -> None:
        self._target.update(self._config.polyak, self._model)

    def select_action(
            self,
            states: np.ndarray,
            noisy_scale: float = 0.0
    ) -> np.ndarray:
        return self._tf_select_action(states, noisy_scale).numpy()

    @cast(Types.Function, tf.function)
    def _tf_select_action(
            self,
            states: Types.Tensor,
            noisy_scale: float,
    ) -> tf.Tensor:
        pi = self._model.pi(states)
        pi += noisy_scale * tf.random.normal(pi.shape)
        pi = tf.clip_by_value(pi, self._config.action_space.low, self._config.action_space.high)

        return pi
