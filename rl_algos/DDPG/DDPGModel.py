import tensorflow as tf


class DDPGModel(object):
    def __init__(
            self,
            pi_model: tf.keras.Model,
            q_model: tf.keras.Model,
            q_pi_model: tf.keras.Model,
    ) -> None:
        self._pi_model = pi_model
        self._q_model = q_model
        self._q_pi_model = q_pi_model

    def pi(
            self,
            states: tf.Tensor,
    ) -> tf.Tensor:
        return self._pi_model(states)

    def q(
            self,
            states: tf.Tensor,
            actions: tf.Tensor,
    ) -> tf.Tensor:
        inputs = tf.concat([states, actions], axis=-1)
        return self._q_model(inputs)

    def q_pi(
            self,
            states: tf.Tensor,
            pi: tf.Tensor,
    ) -> tf.Tensor:
        inputs = tf.concat([states, pi], axis=-1)
        return self._q_pi_model(inputs)
