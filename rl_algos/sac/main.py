import datetime

import gym
import numpy as np
import tensorflow as tf

from rl_algos.sac.SAC import SAC
from rl_algos.sac.SACAgent import SACAgent
from rl_algos.sac.SACConfig import SACConfig
from rl_algos.sac.SACModel import SACModel
from rl_algos.utils.ReplayBuffer import ReplayBuffer


def main() -> None:
    # env_name, reward_scaling_factor, = "Pendulum-v0", 1.0 / 100
    # env_name, reward_scaling_factor = "MountainCarContinuous-v0", 1.0
    # env_name, reward_scaling_factor = "LunarLanderContinuous-v2", 1
    env_name, reward_scaling_factor = "BipedalWalker-v2", 1
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    config = SACConfig(env.observation_space, env.action_space)
    config.reward_scaling_factor = reward_scaling_factor

    model = build_model(config)

    timestamp = str(datetime.datetime.now()).replace(":", " ").replace(".", " ")
    writer = tf.summary.create_file_writer(f"./tmp/ppo/{timestamp}")

    buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], config.buffer_size)
    agent = SACAgent(model, config, writer)

    sac = SAC(env, test_env, agent, buffer, config, writer)
    sac.run()


def build_model(config: SACConfig) -> SACModel:
    return SACModel(build_pi_model(config), build_q_model(config), build_q_model(config), build_v_model(config), build_v_model(config), config)


def build_pi_model(config: SACConfig) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(config.observation_space.shape)
    outputs = inputs

    outputs = tf.keras.layers.Dense(400, activation="relu")(outputs)
    outputs = tf.keras.layers.Dense(300, activation="relu")(outputs)

    mu = tf.keras.layers.Dense(sum(config.action_space.shape))(outputs)
    log_std = tf.keras.layers.Dense(sum(config.action_space.shape))(outputs)

    return tf.keras.Model(inputs=inputs, outputs=[mu, log_std])


def build_v_model(config: SACConfig) -> tf.keras.Model:
    shape = config.observation_space.shape
    inputs = tf.keras.layers.Input(shape)

    outputs = inputs
    outputs = tf.keras.layers.Dense(400, activation="relu")(outputs)
    outputs = tf.keras.layers.Dense(300, activation="relu")(outputs)
    outputs = tf.squeeze(tf.keras.layers.Dense(1, activation="linear")(outputs))

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_q_model(config: SACConfig) -> tf.keras.Model:
    shape = np.concatenate([np.ones(config.observation_space.shape), np.ones(config.action_space.shape)], axis=-1).shape
    inputs = tf.keras.layers.Input(shape)

    outputs = inputs
    outputs = tf.keras.layers.Dense(400, activation="relu")(outputs)
    outputs = tf.keras.layers.Dense(300, activation="relu")(outputs)
    outputs = tf.squeeze(tf.keras.layers.Dense(1, activation="linear")(outputs))

    return tf.keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    main()
