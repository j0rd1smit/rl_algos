from typing import cast, Optional

import numpy as np
import tensorflow as tf

from rl_algos.utils.type_utils import TfFunctionType


class Agent(object):
    def __init__(
            self,
            config: "AgentConfig",
            base_model: tf.keras.Model,
            policy_model: tf.keras.Model,
            value_model: Optional[tf.keras.Model] = None,

    ) -> None:
        self.config = config
        self.base_model = base_model
        self.policy_model = policy_model
        self.value_model = value_model
        self.optimizer = tf.keras.optimizers.Adam(lr=config.lr)
        self.value_optimizer = tf.keras.optimizers.Adam(lr=config.lr)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        return self._select_action(obs).numpy()

    @cast(TfFunctionType, tf.function)
    def _select_action(self, obs: np.ndarray) -> tf.Tensor:
        base = self.base_model(obs)
        pi = self.policy_model(base)
        return tf.random.categorical(pi, 1)

    def training_step(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        self._training_step(states, actions, returns)

    @cast(TfFunctionType, tf.function)
    def _training_step(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        if self.config.base_line:
            returns = returns - tf.reduce_mean(returns)

        if self.value_model is not None:
            with tf.GradientTape() as tape:
                base = self.base_model(states)
                v = self.value_model(base)
                v_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true=returns, y_pred=v))

            value_variable = self.base_model.trainable_variables + self.value_model.trainable_variables
            gradients = tape.gradient(v_loss, value_variable)
            self.value_optimizer.apply_gradients(zip(gradients, value_variable))

        with tf.GradientTape() as tape:
            base = self.base_model(states)
            logits = self.policy_model(base)

            policy = tf.nn.softmax(logits)
            log_policy = tf.nn.log_softmax(logits)

            indices = tf.stack([tf.range(tf.shape(log_policy)[0]), actions], axis=-1)
            log_policy_for_actions = tf.gather_nd(log_policy, indices)

            if self.value_model is not None:
                v = self.value_model(base)
                targets = returns - v
            else:
                targets = returns

            # entropy per example
            actor_loss = -tf.reduce_mean(log_policy_for_actions * tf.stop_gradient(targets))
            entropy_per_observation = tf.reduce_sum(policy * log_policy, axis=1, name="entropy")

            loss = actor_loss - self.config.entropy_weight * entropy_per_observation

        policy_variables = self.base_model.trainable_variables + self.policy_model.trainable_variables
        gradients = tape.gradient(loss, policy_variables)
        self.optimizer.apply_gradients(zip(gradients, policy_variables))




class AgentConfig(object):
    def __init__(
            self,
            lr: float = 1e-3,
            batch_size: int = 32,
            base_line: bool = True,
            entropy_weight: float = 0.1,
    ) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.base_line = base_line
        self.entropy_weight = entropy_weight
