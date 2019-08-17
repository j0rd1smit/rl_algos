from typing import cast

import numpy as np
import tensorflow as tf

import rl_algos.utils.Types as Types
from rl_algos.ddpg.DDPGConfig import DDPGConfig
from rl_algos.ddpg.DDPGModel import DDPGModel


class DDPGAgent(object):
    def __init__(
            self,
            model: DDPGModel,
            target: DDPGModel,
            config: DDPGConfig,
            writer: tf.summary.SummaryWriter,
    ) -> None:
        self._model = model
        self._target = target
        self._config = config
        self._writer = writer

        self._optimizer_pi = tf.keras.optimizers.Adam(lr=config.lr_pi)
        self._optimizer_q = tf.keras.optimizers.Adam(lr=config.lr_q)

    def select_action(
            self,
            states: np.ndarray,
            noise_scale: float = 0.0
    ) -> np.ndarray:
        return self._tf_get_action(states, noise_scale).numpy()

    #@cast(Types.Function, tf.function)
    def _tf_get_action(
            self,
            states: Types.Tensor,
            noise_scale: float
    ) -> tf.Tensor:
        pi = self._model.pi(states)
        pi += noise_scale * tf.random.normal(shape=pi.shape)
        pi = tf.clip_by_value(pi, self._config.action_space.low, self._config.action_space.high)

        return pi

    @cast(Types.Function, tf.function)
    def train_pi(
            self,
            states: Types.Tensor,
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self._pi_loss_function(states)

        variables = self._model.pi_variables
        gradients = tape.gradient(loss, variables)
        self._optimizer_pi.apply_gradients(zip(gradients, variables))


        return loss

    def _pi_loss_function(
            self,
            states: tf.Tensor,
    ) -> tf.Tensor:
        return -tf.reduce_mean(self._model.q_pi(states))

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
            loss = self._q_loss_function(states, actions, rewards, next_states, dones)

        variables = self._model.q_variables
        gradients = tape.gradient(loss, variables)
        self._optimizer_q.apply_gradients(zip(gradients, variables))


        return loss

    def _q_loss_function(
            self,
            states: tf.Tensor,
            actions: tf.Tensor,
            rewards: tf.Tensor,
            next_states: tf.Tensor,
            dones: tf.Tensor,
    ) -> tf.Tensor:
        q_values = self._model.q(states, actions)

        q_pi_targets = tf.stop_gradient(self._target.q_pi(next_states))
        # noinspection PyTypeChecker
        bellman_backups = rewards + self._config.gamma * (1.0 - dones) * q_pi_targets

        # noinspection PyTypeChecker
        return tf.reduce_mean((q_values - bellman_backups) ** 2)

    @cast(Types.Function, tf.function)
    def update_target(self) -> None:
        self._target.update(self._config.polyak, self._model)


