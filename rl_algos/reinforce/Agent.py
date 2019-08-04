from typing import cast

import numpy as np
import tensorflow as tf

from rl_algos.utils.type_utils import TfFunctionType


class Agent(object):
    def __init__(
            self,
            model: tf.keras.Model,
            config: "AgentConfig",
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = tf.keras.optimizers.Adam(lr=config.lr)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return self._select_action(obs).numpy()

    @cast(TfFunctionType, tf.function)
    def _select_action(self, obs: np.ndarray) -> tf.Tensor:
        _, pi = self.model(obs)
        return tf.random.categorical(pi, 1)

    def training_step(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        self._training_step(states, actions, returns)

    @cast(TfFunctionType, tf.function)
    def _training_step(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        with tf.GradientTape() as tape:
            v, pi = self.model(states)

            target = returns - v
            pi_loss = tf.reduce_mean(target * tf.keras.losses.sparse_categorical_crossentropy(actions, pi, from_logits=True))

            value_loss = tf.reduce_mean(tf.keras.losses.MSE(returns, v))

            entroy = -tf.reduce_mean(tf.nn.softmax(pi) * tf.nn.log_softmax(pi))
            loss = pi_loss + value_loss + 0.1 * entroy


        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class AgentConfig(object):
    def __init__(
            self,
            lr: float = 1e-3,
            batch_size: int = 32,
            base_line: bool = True,
    ) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.base_line = base_line
