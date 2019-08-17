import datetime

import gym
import numpy as np
import tensorflow as tf

from rl_algos.ddpg.DDPG import DDPG
from rl_algos.ddpg.DDPGAgent import DDPGAgent
from rl_algos.ddpg.DDPGConfig import DDPGConfig
from rl_algos.ddpg.DDPGModel import DDPGModel
from rl_algos.utils.ReplayBuffer import ReplayBuffer


def main() -> None:
    env = gym.make("MountainCarContinuous-v0")
    #env = gym.make("LunarLanderContinuous-v2")
    config = DDPGConfig(env.observation_space, env.action_space)
    print(env.action_space.low)
    print(env.action_space.high)

    config.reward_scaling_factor = 1.0


    model = _model(config)
    target = _model(config)
    #target.update(1, model)

    timestamp = str(datetime.datetime.now()).replace(":", " ").replace(".", " ")
    writer = tf.summary.create_file_writer(f"./tmp/ddpg/{timestamp}")

    size = int(1e6)
    buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], size)

    agent = DDPGAgent(model, target, config, writer)
    ddpg = DDPG(env, agent, buffer, config, writer)

    ddpg.run()


def _model(config: DDPGConfig) -> DDPGModel:
    return DDPGModel(_pi_model(config), _q_model(config))


def _pi_model(config: DDPGConfig) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(config.observation_space.shape)
    outputs = inputs
    for _ in range(4):
        outputs = tf.keras.layers.Dense(32, activation="relu")(outputs)

    outputs = config.action_space.high * tf.keras.layers.Dense(sum(config.action_space.shape), activation="tanh")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _q_model(config: DDPGConfig) -> tf.keras.Model:
    shape = np.concatenate([np.ones(config.observation_space.shape), np.ones(config.action_space.shape)], axis=-1).shape
    inputs = tf.keras.layers.Input(shape)

    outputs = inputs
    for _ in range(4):
        outputs = tf.keras.layers.Dense(32, activation="tanh")(outputs)
    outputs = tf.keras.layers.Dense(1, activation="linear")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)



if __name__ == '__main__':
    main()
