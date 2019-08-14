from typing import cast

import numpy as np
import tensorflow as tf

from rl_algos.utils.Types import Tensor, Function


class Agent(object):
    def __init__(
            self,
            config: "Config",
            model: tf.keras.Model,
    ) -> None:
        self.config = config
        self.model = model

        self.optimizer = tf.keras.optimizers.Adam(lr=config.lr)
        self.critic_loss_metric = tf.keras.metrics.Mean("critic_loss")
        self.actor_loss_metric = tf.keras.metrics.Mean("actor_loss")
        self.loss_metric = tf.keras.metrics.Mean("loss")
        self.entropy_metric = tf.keras.metrics.Mean("entropy")

    def select_actions(self, states: np.ndarray) -> np.ndarray:
        return self._tf_select_action(states).numpy()

    @cast(Function, tf.function)
    def _tf_select_action(self, states: np.ndarray) -> tf.Tensor:
        _, policy = self.model(states)
        actions = tf.random.categorical(policy, 1)
        return tf.cast(actions, tf.int32)

    @cast(Function, tf.function)
    def training_step(self, states: Tensor, rewards: Tensor, actions: Tensor, dones: Tensor, next_states: Tensor) -> None:
        actions = tf.squeeze(actions)
        rewards = tf.expand_dims(rewards, axis=-1)
        dones = tf.expand_dims(dones, axis=-1)
        with tf.GradientTape() as tape:
            loss = self._loss_function(states, rewards, actions, dones, next_states)

        self.loss_metric(tf.reduce_mean(loss))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def _loss_function(self, states: tf.Tensor, rewards: tf.Tensor, actions: tf.Tensor, dones: tf.Tensor, next_states: tf.Tensor) -> tf.Tensor:
        assert len(actions.shape) == 1
        assert len(dones.shape) == 2
        assert len(rewards.shape) == 2

        value, policy_logits = self.model(states)
        value_next_states, _ = self.model(next_states)
        value_next_states = tf.compat.v2.where(dones, tf.zeros_like(value_next_states), value_next_states)

        probs = tf.nn.softmax(policy_logits)
        log_probs = tf.nn.log_softmax(policy_logits)

        row_indices = tf.range(tf.shape(log_probs)[0])
        indices = tf.stack([row_indices, actions], axis=-1)
        log_policy_for_actions = tf.expand_dims(tf.gather_nd(log_probs, indices), axis=-1)

        # noinspection PyTypeChecker
        target_next_state = rewards + self.config.gamma * value_next_states
        advantage = target_next_state - value
        entropy_per_state = -tf.reduce_mean(probs * log_probs, axis=1)
        actor_loss = -tf.reduce_mean(log_policy_for_actions * tf.stop_gradient(advantage)) - self.config.entropy_weight * entropy_per_state

        critic_errors = tf.stop_gradient(target_next_state) - value
        critic_loss = tf.reduce_mean(critic_errors ** 2)

        loss = critic_loss + actor_loss
        self.entropy_metric(tf.reduce_mean(entropy_per_state))
        self.actor_loss_metric(tf.reduce_mean(actor_loss))
        self.critic_loss_metric(critic_loss)
        return loss


class Config(object):
    def __init__(
            self,
            gamma: float = 0.99,
            lr: float = 1e-4,
            entropy_weight: float = 0.1
    ) -> None:
        self.gamma = gamma
        self.lr = lr
        self.entropy_weight = entropy_weight
