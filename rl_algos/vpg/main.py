from typing import Tuple

import gym
import tensorflow as tf
from gym.spaces import Box

from rl_algos.utils.GAEBuffer import GAEBuffer
from rl_algos.vpg.vpg import VPG
from rl_algos.vpg.VPGAgent import VPGAgent
from rl_algos.vpg.VPGConfig import VPGConfig
import rl_algos.utils.core as core


def main() -> None:
    #env_name = "CartPole-v1"
    env_name = "Pendulum-v0"
    env = gym.make(env_name)

    config = VPGConfig(env.observation_space, env.action_space)
    #policy = core.categorical_policy
    policy = gaussian_policy

    pi_model = build_pi_model(config)
    v_model = build_v_model(config)

    agent = VPGAgent(pi_model, v_model, policy, config)
    buffer = GAEBuffer(env.observation_space.shape, env.action_space.shape, config.gamma, config.lamb, config.steps_per_epoch)

    vpg = VPG(env, agent, buffer, config, render=True)
    vpg.run()

def gaussian_policy(mu: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    log_std = tf.zeros(shape=mu.shape) + 0.1
    return core.gaussian_policy(mu, log_std, actions)

def build_pi_model(config: VPGConfig) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(config.observation_space.shape,  dtype=tf.float32)
    outputs = inputs
    for _ in range(4):
        outputs = tf.keras.layers.Dense(32, activation="relu")(outputs)

    if isinstance(config.action_space, Box):
        outputs = tf.keras.layers.Dense(sum(config.action_space.shape), activation="linear")(outputs)
    else:
        outputs = tf.keras.layers.Dense(sum(config.action_space.n), activation="linear")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_v_model(config: VPGConfig) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(config.observation_space.shape, dtype=tf.float32)

    outputs = inputs
    for _ in range(4):
        outputs = tf.keras.layers.Dense(32, activation="relu")(outputs)

    outputs = tf.keras.layers.Dense(1, activation="linear")(outputs)
    tf.squeeze(outputs, axis=-1)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    main()