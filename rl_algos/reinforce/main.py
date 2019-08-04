from typing import Tuple

import tensorflow as tf
import numpy as np
import gym
import tqdm

from rl_algos.reinforce.Agent import Agent, AgentConfig


def main() -> None:
    gamma = 0.99
    n_episodes = 5000
    example_every = 50

    env = gym.make("CartPole-v1")
    config = AgentConfig()


    shape = env.observation_space.shape
    n_actions = env.action_space.n
    model = build_model(shape, n_actions)
    agent = Agent(model, config)

    for i in tqdm.trange(n_episodes):
        s = env.reset()
        done = False

        obs = []
        actions = []
        rewards = []
        returns = []

        while not done:
            a = agent.select_action(np.array([s])).flatten()[0]
            s_next, r, done, _ = env.step(a)

            obs.append(s)
            actions.append(a)
            rewards.append(r)
            s = s_next

        return_so_far = 0.0
        for r in reversed(rewards):
            return_so_far = gamma * return_so_far + r
            returns.append(return_so_far)

        returns = returns[::-1]

        obs = np.array(obs)
        actions = np.array(actions)
        returns =  np.array(returns)
        agent.training_step(obs, actions, returns)

        if (i + 1) % example_every == 0:
            total_r = 0
            s = env.reset()
            done = False
            while not done:
                a = agent.select_action(np.array([s])).flatten()[0]
                s, r, done, _ = env.step(a)
                total_r += r
            print(total_r)


    env.close()


def build_model(shape: Tuple[int], n_actions: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=shape, dtype=tf.float64)
    x = inputs
    for _ in range(4):
        x = tf.keras.layers.Dense(32, activation="relu")(x)

    pi = tf.keras.layers.Dense(n_actions)(x)
    v = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=[v, pi])
    return model


if __name__ == '__main__':
    main()




