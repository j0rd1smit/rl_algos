from typing import Tuple

import gym
import numpy as np
import tensorflow as tf
import tqdm

from rl_algos.a2c.Agent import Agent, Config
from rl_algos.a2c.BatchEnv import BatchEnv


def main() -> None:
    n_envs = 128
    n_training_steps = 10000
    n_demo_step = 5000
    print_every_n_steps = 1000
    config = Config()

    env_name = "CartPole-v1"
    test_env = gym.make(env_name)
    envs = [gym.make(env_name) for _ in range(n_envs)]
    batch_env = BatchEnv(envs)
    env_shape = envs[0].observation_space.shape
    n_actions = envs[0].action_space.n

    model = build_model(env_shape, n_actions)
    agent = Agent(config, model)

    states = batch_env.reset()
    for i in tqdm.trange(n_training_steps):
        actions = agent.select_actions(states)
        next_states, rewards, dones, _ = batch_env.step(actions)

        agent.training_step(states, rewards, actions, dones, next_states)
        states = next_states

        if (i + 1) % print_every_n_steps == 0:
            done = False
            s = test_env.reset()
            total_r = 0
            while not done:
                a = agent.select_actions(np.array([s]))[0][0]
                s, r, done, _ = test_env.step(a)
                total_r += r
            print(f"total_r: {total_r}")
            print(f"entropy_metric: {agent.entropy_metric.result()}")
            print(f"actor_loss_metric: {agent.actor_loss_metric.result()}")
            print(f"critic_loss_metric: {agent.critic_loss_metric.result()}")
            print(f"loss_metric: {agent.loss_metric.result()}")
            agent.entropy_metric.reset_states()
            agent.actor_loss_metric.reset_states()
            agent.critic_loss_metric.reset_states()
            agent.loss_metric.reset_states()

    s = test_env.reset()
    for _ in tqdm.trange(n_demo_step):
        test_env.render()
        a = agent.select_actions(np.array([s]))[0][0]
        s, r, done, _ = test_env.step(a)
        if done:
            s = test_env.reset()


def build_model(shape: Tuple[int], n_actions: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=shape, dtype=tf.float32)
    x = inputs
    for _ in range(3):
        x = tf.keras.layers.Dense(32, activation="relu")(x)

    v = tf.keras.layers.Dense(32, activation="relu")(x)
    v = tf.keras.layers.Dense(1)(v)

    pi = tf.keras.layers.Dense(32, activation="relu")(x)
    pi = tf.keras.layers.Dense(n_actions)(pi)

    model = tf.keras.Model(inputs=inputs, outputs=[v, pi])
    return model


if __name__ == '__main__':
    main()
