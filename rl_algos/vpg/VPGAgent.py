from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast, Callable

from rl_algos.vpg.VPGConfig import VPGConfig
import tensorflow as tf
import rl_algos.utils.Types as Types



PolicyFunction = Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]


class VPGAgent(object):
    def __init__(
            self,
            pi_model: tf.keras.Model,
            v_model: tf.keras.Model,
            policy_function: PolicyFunction,
            config: VPGConfig,
    ) -> None:
        self._pi_model = pi_model
        self._v_model = v_model
        self._policy_function = policy_function
        self._config = config

        self.optimizer_v = tf.keras.optimizers.Adam(lr=self._config.lr_v)
        self.optimizer_pi = tf.keras.optimizers.Adam(lr=self._config.lr_pi)

    @cast(Types.Function, tf.function)
    def predict_value(
            self,
            states: Types.Tensor,
        ) -> tf.Tensor:
        return self._v_model(states)

    @cast(Types.Function, tf.function)
    def select_action(
            self,
            states: Types.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        model_output = self._pi_model(states)
        shape = (model_output.shape[0],) + self._config.action_space.shape
        pi, _, logp_pi = self._policy_function(model_output, tf.zeros(shape, dtype=tf.int32))
        v = self._v_model(states)

        return pi, v, logp_pi

    @cast(Types.Function, tf.function)
    def train_pi(
            self,
            states: Types.Tensor,
            actions: Types.Tensor,
            advantage: Types.Tensor,
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self._pi_loss(states, actions, advantage)

        variables = self._pi_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer_v.apply_gradients(zip(gradients, variables))

        return loss

    def _pi_loss(
            self,
            states: tf.Tensor,
            actions: tf.Tensor,
            advantage: tf.Tensor,
        ) -> tf.Tensor:
        model_output = self._pi_model(states)
        _, logp, _ = self._policy_function(model_output, actions)

        tf.assert_equal(logp.shape, advantage.shape)

        return -tf.reduce_mean(logp * advantage)

    @cast(Types.Function, tf.function)
    def train_v(
            self,
            states: Types.Tensor,
            returns: Types.Tensor,
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self._v_loss(states, returns)

        variables = self._v_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer_v.apply_gradients(zip(gradients, variables))

        return loss

    def _v_loss(
            self,
            states: tf.Tensor,
            returns: tf.Tensor,
        ) -> tf.Tensor:
        v = self._v_model(states)
        return tf.reduce_mean((returns - v) ** 2)
