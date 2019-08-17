from typing import Any

import tensorflow as tf

import rl_algos.utils.core as core


class DDPGModel(object):
    def __init__(
            self,
            pi_model: tf.keras.Model,
            q_model: tf.keras.Model,
    ) -> None:
        self._pi_model = pi_model
        self._q_model = q_model

    @property
    def pi_variables(self) -> Any:
        return self._pi_model.trainable_variables

    @property
    def q_variables(self) -> Any:
        return self._q_model.trainable_variables

    def pi(
            self,
            states: tf.Tensor,
    ) -> tf.Tensor:
        return self._pi_model(states)

    def q(
            self,
            states: tf.Tensor,
            actions: tf.Tensor,
    ) -> tf.Tensor:
        inputs = tf.concat([states, actions], axis=-1)
        return self._q_model(inputs)

    def q_pi(
            self,
            states: tf.Tensor,
    ) -> tf.Tensor:
        pi = self.pi(states)
        return self.q(states, actions=pi)

    def update(self, polyak: float, other: "DDPGModel") -> None:
        core.polyak_avg_vars(polyak, other._pi_model, self._pi_model)
        core.polyak_avg_vars(polyak, other._q_model, self._q_model)
