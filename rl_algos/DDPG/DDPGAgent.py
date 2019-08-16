import numpy as np
import tensorflow as tf

import rl_algos.utils.Types as Types
from rl_algos.DDPG.DDPGConfig import DDPGConfig
from rl_algos.DDPG.DDPGModel import DDPGModel


def _pi_loss(q_pi: tf.Tensor) -> tf.Tensor:
    return -tf.reduce_mean(q_pi)


def _bellman_backup(
        rewards: tf.Tensor,
        gamma: float,
        dones: tf.Tensor,
        q_pi_targets: tf.Tensor
) -> tf.Tensor:
    # noinspection PyTypeChecker
    return tf.stop_gradient(rewards + gamma * (1 - dones) * q_pi_targets)


def _q_loss(
        q: tf.Tensor,
        rewards: tf.Tensor,
        gamma: float,
        dones: tf.Tensor,
        q_pi_targets: tf.Tensor
) -> tf.Tensor:
    backup = _bellman_backup(rewards, gamma, dones, q_pi_targets)
    return tf.reduce_mean((q - backup) ** 2)


class DDPGAgent(object):
    def __init__(
            self,
            model: DDPGModel,
            target_model: DDPGModel,
            config: DDPGConfig,
    ) -> None:
        self._model = model
        self._target_model = target_model

        self._config = config

    def get_action(
            self,
            states: np.ndarray,
    ) -> np.ndarray:
        return self._tf_get_action(states).numpy()

    # tf.function
    def _tf_get_action(
            self,
            states: Types.Tensor
    ) -> tf.Tensor:
        pi = self._model.pi(states)
        pi += self._config.noise_scale * tf.random.normal(shape=pi.shape)
        pi = tf.clip_by_value(pi, self._config.action_space.low, self._config.action_space.high)
        return pi

    def _q_loss_function(
            self,
            states: tf.Tensor,
            actions: tf.Tensor,
            rewards: tf.Tensor,
            next_states: tf.Tensor,
            dones: tf.Tensor,
    ) -> tf.Tensor:
        q_values = self._model.q(states, actions)
        q_pi_targets = self._target_model.q_pi(next_states, actions)

        return _q_loss(q_values, rewards, self._config.gamma, dones, q_pi_targets)

    def _pi_loss_function(
            self,
            states: tf.Tensor,
    ) -> tf.Tensor:
        pi = self._model.pi(states)
        q_pi = self._model.q_pi(states, pi)
        return _pi_loss(q_pi)
