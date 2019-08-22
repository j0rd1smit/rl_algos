from typing import cast, Tuple

import numpy as np
import tensorflow as tf

import rl_algos.utils.Types as Types
from rl_algos.sac.SACConfig import SACConfig
from rl_algos.sac.SACModel import SACModel


class SACAgent(object):
    def __init__(
            self,
            sac_model: SACModel,
            config: SACConfig,
            writer: tf.summary.SummaryWriter,
    ) -> None:
        self._sac_model = sac_model
        self._config = config
        self._writer = writer

        self._optimizer_v = tf.keras.optimizers.Adam(lr=self._config.lr_v)
        self._optimizer_pi = tf.keras.optimizers.Adam(lr=self._config.lr_pi)

    def select_actions(
            self,
            states: Types.Tensor,
            deterministic: bool,
    ) -> np.ndarray:
        return self._tf_select_actions(states, deterministic).numpy()

    @cast(Types.Function, tf.function)
    def _tf_select_actions(
            self,
            states: Types.Tensor,
            deterministic: bool,
    ) -> tf.Tensor:
        mu, pi, _ = self._sac_model.pi(states)
        if deterministic:
            return mu
        return pi

    @cast(Types.Function, tf.function)
    def train(
            self,
            states: Types.Tensor,
            actions: Types.Tensor,
            rewards: Types.Tensor,
            next_states: Types.Tensor,
            dones: Types.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        pi_loss = self._train_actor(states)
        v_loss = self._train_critic(states, actions, rewards, next_states, dones)

        self._sac_model.update_target()

        return pi_loss, v_loss

    def _train_actor(
            self,
            states: tf.Tensor,
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self._pi_loss(states)

        variables = self._sac_model.actor_variables
        gradients = tape.gradient(loss, variables)
        self._optimizer_pi.apply_gradients(zip(gradients, variables))

        return loss

    def _pi_loss(
            self,
            states: tf.Tensor,
    ) -> tf.Tensor:
        mu, pi, logp_pi = self._sac_model.pi(states)
        q1_pi = self._sac_model.q1(states, pi)
        pi_loss = tf.reduce_mean(self._config.entropy_regularization_weight * logp_pi - q1_pi)
        return pi_loss

    def _train_critic(
            self,
            states: Types.Tensor,
            actions: Types.Tensor,
            rewards: Types.Tensor,
            next_states: Types.Tensor,
            dones: Types.Tensor,
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self._v_loss(states, actions, rewards, next_states, dones)

        variables = self._sac_model.critic_variables
        gradients = tape.gradient(loss, variables)
        self._optimizer_v.apply_gradients(zip(gradients, variables))

        return loss

    def _v_loss(
            self,
            states: Types.Tensor,
            actions: Types.Tensor,
            rewards: Types.Tensor,
            next_states: Types.Tensor,
            dones: Types.Tensor,
    ) -> tf.Tensor:
        _, pi, logp_pi = self._sac_model.pi(states)
        q1 = self._sac_model.q1(states, actions)
        q2 = self._sac_model.q2(states, actions)
        v = self._sac_model.v(states)
        q1_pi = self._sac_model.q1(states, pi)
        q2_pi = self._sac_model.q2(states, pi)
        v_target = self._sac_model.v_target(next_states)
        min_q_pi = tf.minimum(q1_pi, q2_pi)

        q_backup = tf.stop_gradient(rewards + self._config.gamma * (1.0 - dones) * v_target)
        v_backup = tf.stop_gradient(min_q_pi - self._config.entropy_regularization_weight * logp_pi)

        q1_loss = tf.keras.losses.Huber()(q_backup, q1)  # 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
        q2_loss = tf.keras.losses.Huber()(q_backup, q2)  # 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
        v_loss = tf.keras.losses.Huber()(v_backup, v)  # 0.5 * tf.reduce_mean((v_backup - v) ** 2)

        loss = q1_loss + q2_loss + v_loss
        return loss
