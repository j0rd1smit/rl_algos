import gym
import tensorflow as tf

from rl_algos.DDPG.DDPG import DDPG
from rl_algos.DDPG.DDPGAgent import DDPGAgent
from rl_algos.DDPG.DDPGConfig import DDPGConfig
from rl_algos.DDPG.DDPGModel import DDPGModel


def main() -> None:
    env = gym.make("MountainCarContinuous-v0")
    config = DDPGConfig(env.observation_space, env.action_space)
    model = _ddpg_model(config)
    target_model = _ddpg_model(config)

    agent = DDPGAgent(model, target_model, config)
    ddpg = DDPG(env, agent, config)

    ddpg.train(1000)


def _ddpg_model(config: DDPGConfig) -> DDPGModel:
    return DDPGModel(_pi_model(config), _q_model(), _q_pi_model())


def _pi_model(config: DDPGConfig) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(config.observation_space.shape)
    outputs = inputs
    for _ in range(3):
        outputs = tf.keras.layers.Dense(32, activation="relu")(outputs)

    outputs = config.action_space.high * tf.keras.layers.Dense(sum(config.action_space.shape), activation="tanh")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _q_model() -> tf.keras.Model:
    model = tf.keras.Sequential()
    for _ in range(3):
        model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    return model


def _q_pi_model() -> tf.keras.Model:
    model = tf.keras.Sequential()
    for _ in range(3):
        model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    return model


if __name__ == '__main__':
    main()
