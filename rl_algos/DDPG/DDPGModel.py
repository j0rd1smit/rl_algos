from typing import Any

import tensorflow as tf


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
        inputs = tf.concat([states, pi], axis=-1)
        return self._q_model(inputs)

    def update(self, polyak: float, other: "DDPGModel") -> None:
        polyak_avg_vars(polyak, other._pi_model, self._pi_model)
        polyak_avg_vars(polyak, other._q_model, self._q_model)


def polyak_avg_vars(polyak: float, main: tf.keras.Model, target: tf.keras.Model) -> None:
    assert len(main.trainable_variables) == len(target.trainable_variables)

    for v_main, v_targ in zip(main.trainable_variables, target.trainable_variables):
        updated_value = polyak * v_targ + (1 - polyak) * v_main
        v_targ.assign(updated_value)
