from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Optional, Union, Any, cast
import tensorflow as tf
import numpy as np
import rl_algos.utils.core as core
from rl_algos.trpo.TRPOConfig import TRPOConfig

def _logp(logits: tf.Tensor) -> tf.Tensor:
    return tf.nn.log_softmax(logits)

def _logp_per_action(logp: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
    return core.select_value_per_action(logp, actions)

def _ratio(logp_per_action: tf.Tensor, logp_per_action_old: tf.Tensor) -> tf.Tensor:
    core.assert_same_shape(logp_per_action, logp_per_action_old)
    return tf.exp(logp_per_action - logp_per_action_old)

def _pi_loss(ratio: tf.Tensor, advantage: tf.Tensor) -> tf.Tensor:
    core.assert_same_shape(ratio, advantage)
    return - tf.reduce_mean(ratio * advantage)

def _value_loss(value: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    core.assert_same_shape(value, returns)
    return tf.reduce_mean((value - returns) ** 2) #TODO Try hubert loss

def _hessian_vector_product(
        self,
        f: tf.Tensor,
        param
    ) -> None:
    pass

def _categorical_kl(
        logp0: tf.Tensor,
        logp1: tf.Tensor,
    ) -> tf.Tensor:
    all_kls = tf.reduce_sum(tf.exp(logp1) * (logp1 - logp0), axis=1)
    return tf.reduce_mean(all_kls)

class TRPOAgent(object):
    def __init__(
            self,
            base_model: tf.keras.Model,
            actor_model: tf.keras.Model,
            value_model: tf.keras.Model,
            config: TRPOConfig,
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

    def _tf_train_actor(
            self,
            obs: tf.Tensor,
            actions: tf.Tensor,
            advantages: tf.Tensor,
            logp_per_action_old: tf.Tensor,
            logp_all_old: tf.Tensor,
        ) -> None:
        with tf.GradientTape() as tape:
            loss, d_kl = self._pi_loss_function(obs, actions, advantages, logp_per_action_old, logp_all_old)

        variables = self._base_model.trainable_variables + self._actor_model.trainable_variables
        gradients = tape.gradient(loss, variables) #Not sure if is already flat
        assert len(gradients.shape) == 1


    def _pi_loss_function(
            self,
            obs: tf.Tensor,
            actions: tf.Tensor,
            advantages: tf.Tensor,
            logp_per_action_old: tf.Tensor,
            logp_all_old: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        base = self._base_model(obs)
        logits = self._actor_model(base)
        logp = _logp(logits)
        logp_per_action = _logp_per_action(logp, actions)
        ratio = _ratio(logp_per_action, logp_per_action_old)
        d_kl = _categorical_kl(logp, logp_all_old)

        return _pi_loss(ratio, advantages), d_kl
 