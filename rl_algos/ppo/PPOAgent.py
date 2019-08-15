from typing import cast, Tuple

import numpy as np
import tensorflow as tf

import rl_algos.utils.core as core
import rl_algos.utils.Types as Types


class PPOAgent(object):

    def __init__(
            self,
            base_model: tf.keras.Model,
            actor_model: tf.keras.Model,
            value_model: tf.keras.Model,
            config: "Config",
            writer: tf.summary.SummaryWriter,
    ) -> None:
        self._base_model = base_model
        self._actor_model = actor_model
        self._value_model = value_model
        self._config = config
        self._actor_optimizer = tf.keras.optimizers.Adam(lr=self._config.pi_lr)
        self._critic_optimizer = tf.keras.optimizers.Adam(lr=self._config.vf_lr)
        self._writer = writer
        self._step = 0

    def predict_value(self, states: np.ndarray) -> np.ndarray:
        return self._tf_predict_value(states).numpy()

    @cast(Types.Function, tf.function)
    def _tf_predict_value(self, states: Types.Tensor) -> tf.Tensor:
        base = self._base_model(states)
        values = self._value_model(base)
        return values

    def select_actions(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions, value, logp = self._tf_select_action(states)
        return actions.numpy(), value.numpy(), logp.numpy()

    @cast(Types.Function, tf.function)
    def _tf_select_action(self, states: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        base = self._base_model(states)
        logits = self._actor_model(base)
        actions = tf.cast(tf.random.categorical(logits, 1), self._config.tf_int)

        value = self._value_model(base)
        logp_per_action = self.log_p_per_actions(logits, actions)

        return actions, value, logp_per_action

    def train(
            self,
            obs: Types.Tensor,
            actions: Types.Tensor,
            advantages: Types.Tensor,
            returns: Types.Tensor,
            logp_old: Types.Tensor
    ) -> None:
        pi_loss_old, value_loss_old, _, _, _ = self._tf_stats(obs, actions, advantages, returns, logp_old)

        for _ in range(self._config.train_v_iters):
            self.tf_train_critic(obs, returns)

        for i in range(self._config.train_pi_iters):
            kl = self.tf_train_actor(obs, actions, advantages, logp_old).numpy()
            if kl > 1.5 * self._config.target_kl:
                print(f"Breaked pi training at {i}/{self._config.train_pi_iters}  with kl={kl}")
                break



        pi_loss, value_loss, kl, approx_ent, clipfrac = self._tf_stats(obs, actions, advantages, returns, logp_old)
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
        self._writer.flush()
        self._step += 1




    @cast(Types.Function, tf.function)
    def tf_train_actor(
            self,
            obs: Types.Tensor,
            actions: Types.Tensor,
            advantages: Types.Tensor,
            logp_old: Types.Tensor
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            base = self._base_model(obs)
            logits = self._actor_model(base)
            log_p = tf.squeeze(self.log_p_per_actions(logits, actions))
            pi_loss = self.pi_loss_function(log_p, advantages, logp_old)

        variables = self._base_model.trainable_variables + self._actor_model.trainable_variables
        gradients = tape.gradient(pi_loss, variables)
        self._actor_optimizer.apply_gradients(zip(gradients, variables))

        return self.approx_kl(logp_old, log_p)

    def log_p_per_actions(
            self,
            logits: tf.Tensor,
            actions: tf.Tensor,
    ) -> tf.Tensor:
        logp = tf.nn.log_softmax(logits)
        logp_per_action = tf.expand_dims(core.select_value_per_action(logp, actions), axis=-1)
        return logp_per_action

    def pi_loss_function(
            self,
            logp: tf.Tensor,
            advantages: tf.Tensor,
            logp_old: tf.Tensor
    ) -> tf.Tensor:
        core.assert_same_shape(logp, logp_old)
        core.assert_same_shape(logp, advantages)

        clip_ratio = self._config.clip_ratio
        ratio = tf.exp(logp - logp_old)
        # noinspection PyTypeChecker
        min_adv = tf.where(advantages >= 0, (1 + clip_ratio) * advantages, (1 - clip_ratio) * advantages)
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))
        return pi_loss

    def approx_kl(
            self,
            logp_old: tf.Tensor,
            logp: tf.Tensor,
    ) -> tf.Tensor:
        core.assert_same_shape(logp_old, logp_old)
        return tf.reduce_mean(logp_old - logp)

    @cast(Types.Function, tf.function)
    def tf_train_critic(
            self,
            obs: Types.Tensor,
            returns: Types.Tensor,
    ) -> None:
        with tf.GradientTape() as tape:
            base = tf.stop_gradient(self._base_model(obs))
            values = self._value_model(base)
            value_loss = self.value_loss_function(values, returns)

        variables = self._base_model.trainable_variables + self._value_model.trainable_variables
        gradients = tape.gradient(value_loss, variables)
        self._actor_optimizer.apply_gradients(zip(gradients, variables))

    def value_loss_function(
            self,
            values: tf.Tensor,
            returns: tf.Tensor
    ) -> tf.Tensor:
        if len(values.shape) > 1:
            values = tf.squeeze(values)
        core.assert_same_shape(values, returns)
        value_loss = tf.reduce_mean((returns - values) ** 2)

        return value_loss

    def _tf_stats(
            self,
            obs: Types.Tensor,
            actions: Types.Tensor,
            advantages: Types.Tensor,
            returns: Types.Tensor,
            logp_old: Types.Tensor,
    ) -> Tuple[Types.Tensor, Types.Tensor, Types.Tensor, Types.Tensor, Types.Tensor,]:
        base = self._base_model(obs)
        logits = self._actor_model(base)
        values = self._value_model(base)
        log_p = tf.squeeze(self.log_p_per_actions(logits, actions))

        pi_loss = self.pi_loss_function(log_p, advantages, logp_old)
        value_loss = self.value_loss_function(values, returns)
        kl = self.approx_kl(logp_old, log_p)
        approx_ent = tf.reduce_mean(-log_p)
        ratio = tf.exp(log_p - logp_old)
        clipped = tf.logical_or(ratio > (1 + self._config.clip_ratio), ratio < (1 - self._config.clip_ratio))
        clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        return pi_loss, value_loss, kl, approx_ent, clipfrac



class Config(object):
    def __init__(
            self
    ) -> None:
        self.pi_lr = 3e-4
        self.vf_lr = 1e-3
        self.train_pi_iters = 80
        self.train_v_iters = 80
        self.clip_ratio = 0.1
        self.target_kl = 0.005
        self.tf_int = tf.int32
        self.tf_float = tf.float32
