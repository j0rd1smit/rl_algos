from typing import List, Tuple

import tensorflow as tf

import rl_algos.utils.core as core
from rl_algos.sac.SACConfig import SACConfig


class SACModel(object):
    def __init__(
            self,
            pi_model: tf.keras.Model,
            q1_model: tf.keras.Model,
            q2_model: tf.keras.Model,
            v_model: tf.keras.Model,
            v_target_model: tf.keras.Model,
            config: SACConfig,
            log_std_min: float = -20,
            log_std_max: float = 2
    ) -> None:
        self._pi_model = pi_model
        self._q1_model = q1_model
        self._q2_model = q2_model
        self._v_model = v_model
        self._v_target_model = v_target_model
        self._config = config
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

    @property
    def actor_variables(self) -> List[tf.Tensor]:
        return self._pi_model.trainable_variables

    @property
    def critic_variables(self) -> List[tf.Tensor]:
        return self._q1_model.trainable_variables + self._q2_model.trainable_variables + self._v_model.trainable_variables

    def pi(
            self,
            states: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        mu, log_std = self._pi_model(states)

        log_std = tf.nn.tanh(log_std)
        log_std = self._log_std_min + 0.5 * (self._log_std_max - self._log_std_min) * (log_std + 1)  # differantiable reward clipping
        std = tf.exp(log_std)

        pi = mu + tf.random.normal(mu.shape) * std
        log_pi = core.gaussian_likelihood(pi, mu, log_std)
        mu, pi, log_pi = apply_squashing_func(mu, pi, log_pi)
        mu = mu * self._config.action_space.high
        pi = pi * self._config.action_space.high

        return mu, pi, log_pi

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

    def v(
            self,
            states: tf.Tensor,
    ) -> tf.Tensor:
        return self._v_model(states)

    def v_target(
            self,
            states: tf.Tensor,
    ) -> tf.Tensor:
        return self._v_target_model(states)

    def update_target(
            self,
    ) -> None:
        core.polyak_avg_vars(self._config.polyak, main=self._v_model, target=self._v_target_model)


# noinspection PyTypeChecker
def apply_squashing_func(mu: tf.Tensor, pi: tf.Tensor, logp_pi: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.math.log(clip_but_pass_gradient(1 - pi ** 2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


# noinspection PyTypeChecker
def clip_but_pass_gradient(x: tf.Tensor, l: float = -1., u: float = 1.) -> tf.Tensor:
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x) * clip_up + (l - x) * clip_low)


def _combine_state_and_action(
        states: tf.Tensor,
        actions: tf.Tensor
) -> tf.Tensor:
    return tf.concat([states, actions], axis=-1)
