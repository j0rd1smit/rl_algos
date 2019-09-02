import datetime

import gym
import numpy as np
import tensorflow as tf

from rl_algos.td3.TD3 import TD3
from rl_algos.td3.TD3Agent import TD3Agent
from rl_algos.td3.TD3Config import TD3Config
from rl_algos.td3.TD3Model import TD3Model
from rl_algos.utils.ReplayBuffer import ReplayBuffer


def main() -> None:
    #env_name, reward_scaling_factor, = "Pendulum-v0", 1.0 / 100
    #env_name, reward_scaling_factor = "MountainCarContinuous-v0", 1.0
    env_name, reward_scaling_factor = "LunarLanderContinuous-v2", 1.0
    env = gym.make(env_name)
    config = TD3Config(env.observation_space, env.action_space)
    config.reward_scaling_factor = reward_scaling_factor

    live_model = build_model(config)
    target_model = build_model(config)

    timestamp = str(datetime.datetime.now()).replace(":", " ").replace(".", " ")
    writer = tf.summary.create_file_writer(f"./tmp/td3/{timestamp}")

    buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], config.buffer_size)

    agent = TD3Agent(live_model, target_model, config)

    td3 = TD3(env, agent, buffer, config, writer)
    td3.run()


def build_model(config: TD3Config) -> TD3Model:
    return TD3Model(build_pi_model(config), build_q_model(config), build_q_model(config))


def build_pi_model(config: TD3Config) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(config.observation_space.shape)
    outputs = inputs
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(400, activation="relu")(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(300, activation="relu")(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)

    outputs = config.action_space.high * tf.keras.layers.Dense(sum(config.action_space.shape), activation="tanh")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_q_model(config: TD3Config) -> tf.keras.Model:
    shape = np.concatenate([np.ones(config.observation_space.shape), np.ones(config.action_space.shape)], axis=-1).shape
    inputs = tf.keras.layers.Input(shape)

    outputs = inputs
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(400, activation="relu")(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(300, activation="relu")(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(1, activation="linear")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    main()
