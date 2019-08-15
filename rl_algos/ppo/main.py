import datetime
from typing import Tuple

import gym
import tensorflow as tf

from rl_algos.ppo.PPO import PPO, PPOConfig
from rl_algos.ppo.PPOAgent import Config, PPOAgent
from rl_algos.ppo.PPOBuffer import PPOBuffer


def main() -> None:
    gamma = 0.99
    lamd = 0.97
    epochs = 50
    steps_per_epoch = 4000

    agent_config = Config()
    ppo_config = PPOConfig()
    ppo_config.steps_per_epoch = steps_per_epoch

    buffer_size = steps_per_epoch

    env_name = "CartPole-v1"
    env = gym.make(env_name)

    env_shape = env.observation_space.shape
    n_actions = env.action_space.n
    action_shape = (1, )

    timestamp = str(datetime.datetime.now()).replace(":", " ").replace(".", " ")
    writer = tf.summary.create_file_writer(f"./tmp/ppo/{timestamp}")



    bufffer = PPOBuffer(env_shape, action_shape, gamma, lamd, buffer_size)

    base = base_network(env_shape)
    base_output_shape = base.output_shape[1:]
    actor = actor_network(base_output_shape, n_actions)
    critic = critic_network(base_output_shape)

    agent = PPOAgent(base, actor, critic, agent_config, writer)
    ppo = PPO(ppo_config, agent, bufffer, env, writer, render=True)

    ppo.train(epochs)




def base_network(env_shape: Tuple[int]) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=env_shape, dtype=tf.float32)

    x = inputs
    for _ in range(3):
        x = tf.keras.layers.Dense(32, activation="relu")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def critic_network(shape: Tuple[int]) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=shape, dtype=tf.float32)

    x = inputs
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def actor_network(shape: Tuple[int], action_space: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=shape, dtype=tf.float32)

    x = inputs
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(action_space)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

if __name__ == '__main__':
    main()