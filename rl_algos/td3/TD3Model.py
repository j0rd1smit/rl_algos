from typing import List

import tensorflow as tf

import rl_algos.utils.core as core


class TD3Model(object):
    def __init__(
            self,
            pi_model: tf.keras.Model,
            q1_model: tf.keras.Model,
            q2_model: tf.keras.Model,
    ) -> None:
        self._pi_model = pi_model
        self._q1_model = q1_model
        self._q2_model = q2_model

    @property
    def pi_variables(self) -> List[tf.Tensor]:
        return self._pi_model.trainable_variables

    @property
    def q1_variables(self) -> List[tf.Tensor]:
        return self._q1_model.trainable_variables

    @property
    def q2_variables(self) -> List[tf.Tensor]:
        return self._q2_model.trainable_variables

    def pi(
            self,
            states: tf.Tensor,
    ) -> tf.Tensor:
        return self._pi_model(states)

    def q1(
            self,
            states: tf.Tensor,
            actions: tf.Tensor,
    ) -> tf.Tensor:
        return self._q1_model(_combine_state_and_action(states, actions))

    def q2(
            self,
            states: tf.Tensor,
            actions: tf.Tensor,
    ) -> tf.Tensor:
        return self._q2_model(_combine_state_and_action(states, actions))

    def update(self, polyak: float, other: "TD3Model") -> None:
        core.polyak_avg_vars(polyak, other._pi_model, self._pi_model)
        core.polyak_avg_vars(polyak, other._q1_model, self._q1_model)
        core.polyak_avg_vars(polyak, other._q2_model, self._q2_model)


def _combine_state_and_action(
        states: tf.Tensor,
        actions: tf.Tensor
) -> tf.Tensor:
    return tf.concat([states, actions], axis=-1)
