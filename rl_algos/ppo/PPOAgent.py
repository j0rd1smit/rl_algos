from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast

from rl_algos.ppo.PPOConfig import PPOConfig
from rl_algos.utils import Types
import tensorflow as tf
import numpy as np
import rl_algos.utils.core as core


class PPOAgent(object):
    def __init__(
            self,
            pi_model: tf.keras.Model,
            v_model: tf.keras.Model,
            policy_function: Types.PolicyFunction,
            config: PPOConfig,
            writer: tf.summary.SummaryWriter,
    ) -> None:
        self._pi_model = pi_model
        self._v_model = v_model
        self._policy_function = policy_function
        self._config = config
        self._writer = writer

        self._step = 0

        self._optimizer_v = tf.keras.optimizers.Adam(lr=self._config.lr_v)
        self._optimizer_pi = tf.keras.optimizers.Adam(lr=self._config.lr_pi)

        self._huber = tf.keras.losses.Huber()


    def predict_values(
            self,
            states: np.ndarray,
        ) -> np.ndarray:
        return self._tf_predict_values(states).numpy()

    @cast(Types.Function, tf.function)
    def _tf_predict_values(
            self,
            states: Types.Tensor
    ) -> Types.Tensor:
        return self._v_model(states)

    def select_actions(
            self,
            states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pi, v, logp_pi = self._tf_select_actions(states)

        return pi.numpy(), v.numpy(), logp_pi.numpy()

    @cast(Types.Function, tf.function)
    def _tf_select_actions(
            self,
            states: Types.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        model_output = self._pi_model(states)
        shape = (model_output.shape[0],) + self._config.action_space.shape
        pi, _, logp_pi = self._policy_function(model_output, tf.zeros(shape, dtype=tf.int32))
        v = self._v_model(states)

        return pi, v, logp_pi

    def train(
            self,
            states: Types.Tensor,
            actions: Types.Tensor,
            advantages: Types.Tensor,
            returns: Types.Tensor,
            logp_old: Types.Tensor,
        ) -> None:
        pi_loss_old, value_loss_old, _, _, _ = self._tf_stats(states, actions, advantages, returns, logp_old)

        for _ in range(self._config.train_v_iters):
            self._tf_train_critic(states, returns)

        for i in range(self._config.train_pi_iters):
            kl = self._tf_train_actor(states, actions, advantages, logp_old).numpy()
            if kl > 1.5 * self._config.target_kl:
                print(f"Breaked pi training at {i}/{self._config.train_pi_iters}  with kl={kl}")
                break

        pi_loss, value_loss, kl, approx_ent, clipfrac = self._tf_stats(states, actions, advantages, returns, logp_old)
        delta_pi_loss = pi_loss - pi_loss_old
        delta_value_loss = value_loss - value_loss_old
        with self._writer.as_default():
            tf.summary.scalar("pi_loss", pi_loss, step=self._step)
            tf.summary.scalar("value_loss", value_loss, step=self._step)
            tf.summary.scalar("delta_pi_loss", delta_pi_loss, step=self._step)
            tf.summary.scalar("delta_value_loss", delta_value_loss, step=self._step)
            tf.summary.scalar("kl", kl, step=self._step)
            tf.summary.scalar("approx_ent", approx_ent, step=self._step)
            tf.summary.scalar("clipfrac", clipfrac, step=self._step)
            self._step += 1

    @cast(Types.Function, tf.function)
    def _tf_stats(
            self,
            states: Types.Tensor,
            actions: Types.Tensor,
            advantages: Types.Tensor,
            returns: Types.Tensor,
            logp_old: Types.Tensor,
    ) -> Tuple[Types.Tensor, Types.Tensor, Types.Tensor, Types.Tensor, Types.Tensor]:
        pi, logp, logp_pi = self._policy_function(self._pi_model(states), actions)

        value_loss = self._v_loss(states, returns)
        pi_loss, kl = self._pi_loss(states, actions, advantages, logp_old)
        approx_ent = -tf.reduce_mean(logp)
        ratio = tf.exp(logp - logp_old)
        clipped = tf.logical_or(ratio > (1 + self._config.clip_ratio), ratio < (1 - self._config.clip_ratio))
        clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        return pi_loss, value_loss, kl, approx_ent, clipfrac


    @cast(Types.Function, tf.function)
    def _tf_train_actor(
            self,
            states: Types.Tensor,
            actions: Types.Tensor,
            advantages: Types.Tensor,
            logp_old: Types.Tensor
        ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss, approx_kl = self._pi_loss(states, actions, advantages, logp_old)

        variables = self._pi_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self._optimizer_pi.apply_gradients(zip(gradients, variables))

        return approx_kl

    def _pi_loss(
            self,
            states: tf.Tensor,
            actions: tf.Tensor,
            advantages: tf.Tensor,
            logp_old: tf.Tensor,
        ) -> Tuple[tf.Tensor, tf.Tensor]:

        model_output = self._pi_model(states)
        pi, logp, logp_pi = self._policy_function(model_output, actions)

        tf.assert_equal(pi.shape, actions.shape)
        tf.assert_equal(logp.shape, logp_old.shape)
        tf.assert_equal(logp.shape, advantages.shape)

        clip_ratio = self._config.clip_ratio
        ratio = tf.exp(logp - logp_old)
        # noinspection PyTypeChecker
        min_adv = tf.where(advantages >= 0, (1 + clip_ratio) * advantages, (1 - clip_ratio) * advantages)
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))

        # noinspection PyTypeChecker
        return pi_loss, core.approx_kl(logp_old, logp)
       



    @cast(Types.Function, tf.function)
    def _tf_train_critic(
            self,
            states: tf.Tensor,
            returns: tf.Tensor,
    ) -> None:
        with tf.GradientTape() as tape:
            loss = self._v_loss(states, returns)

        variables = self._v_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self._optimizer_pi.apply_gradients(zip(gradients, variables))

    def _v_loss(
            self,
            states: tf.Tensor,
            returns: tf.Tensor,
    ) -> tf.Tensor:
        v = self._v_model(states)
        # TODO return tf.reduce_mean((returns - v) ** 2)
        return self._huber(returns, v)


 