import datetime
from typing import Tuple

import gym
import tensorflow as tf
from gym.spaces import Box

import rl_algos.utils.core as core
from rl_algos.envs.CardGame import CardGame
from rl_algos.envs.TicTacToEnv import TicTacToEnv
from rl_algos.ppo.PPO import PPO
from rl_algos.ppo.PPOAgent import PPOAgent
from rl_algos.ppo.PPOConfig import PPOConfig
from rl_algos.utils.GAEBuffer import GAEBuffer




def main() -> None:
    env_name, reward_scaling_factor, policy = "CartPole-v1", 1.0,  core.categorical_policy
    env_name, reward_scaling_factor, policy = "TicTacTo-v0", 1.0,  core.categorical_policy
    #env_name, reward_scaling_factor, policy = "Pendulum-v0", 1.0, gaussian_policy
    #env_name, reward_scaling_factor, policy = "MountainCarContinuous-v0", 1.0, gaussian_policy
    #env_name, reward_scaling_factor, policy = "LunarLanderContinuous-v2", 1.0, gaussian_policy
    #env_name, reward_scaling_factor, policy = "LunarLander-v2", 1.0/ 100, core.categorical_policy
    # env_name, reward_scaling_factor, policy = "BipedalWalker-v2", 1.0, gaussian_policy
    #env = gym.make(env_name)
    env = CardGame()

    config = PPOConfig(env.observation_space, env.action_space)
    config.reward_scaling_factor = reward_scaling_factor

    pi_model = build_pi_model(config)
    v_model = build_v_model(config)

    timestamp = str(datetime.datetime.now()).replace(":", " ").replace(".", " ")
    writer = tf.summary.create_file_writer(f"./tmp/ppo/{timestamp}")

    agent = PPOAgent(pi_model, v_model, policy, config, writer)
    buffer = GAEBuffer(env.observation_space.shape, env.action_space.shape, config.gamma, config.lamb, config.steps_per_epoch)

    ppo = PPO(env, agent, buffer, config, writer, render=True)
    ppo.run()


def gaussian_policy(mu: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    log_std = tf.zeros(shape=mu.shape) + 0.5
    return core.gaussian_policy(mu, log_std, actions)

def build_pi_model(config: PPOConfig) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(config.observation_space.shape,  dtype=tf.float32)
    outputs = inputs
    outputs = tf.keras.layers.Dense(400, activation="relu")(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(300, activation="relu")(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)

    if isinstance(config.action_space, Box):
        outputs = tf.keras.layers.Dense(sum(config.action_space.shape), activation="linear")(outputs)
    else:
        outputs = tf.keras.layers.Dense(config.action_space.n, activation="linear")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_v_model(config: PPOConfig) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(config.observation_space.shape, dtype=tf.float32)

    outputs = inputs
    outputs = tf.keras.layers.Dense(400, activation="relu")(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(300, activation="relu")(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dense(1, activation="linear")(outputs)

    tf.squeeze(outputs, axis=-1)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    main()