from typing import Tuple

import gym
import numpy as np
import tensorflow as tf
import tqdm

from rl_algos.reinforce.Agent import Agent, AgentConfig


def main() -> None:
    gamma = 0.99
    n_episodes = 500
    example_every = 50
    base_output_shape = (32,)

    env = gym.make("CartPole-v1")
    config = AgentConfig()
    config.base_line = True


    shape = env.observation_space.shape
    n_actions = env.action_space.n
    base_model = build_base_model(shape)
    policy_model = build_policy_model(base_output_shape, n_actions)
    # value_model = build_value_model(base_output_shape)
    value_model = None
    agent = Agent(config, base_model, policy_model, value_model)

    total_rewards = []

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
        total_rewards.append(sum(rewards))

        obs = np.array(obs)
        actions = np.array(actions, dtype=np.int32)
        returns =  np.array(returns)
        agent.training_step(obs, actions, returns)

        if (i + 1) % example_every == 0:
            avg_reward = np.average(total_rewards)
            print(avg_reward)
            if avg_reward > 300:
                break
            total_rewards = []

    s = env.reset()
    for _ in range(1000):
        env.render()
        a = agent.select_action(np.array([s])).flatten()[0]
        s, _, done, _ = env.step(a)
        if done:
            s = env.reset()

    env.close()


def build_base_model(shape: Tuple[int]) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=shape, dtype=tf.float64)
    x = inputs
    for _ in range(4):
        x = tf.keras.layers.Dense(32, activation="relu")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def build_policy_model(shape: Tuple[int], n_actions: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=shape, dtype=tf.float64)
    pi = tf.keras.layers.Dense(n_actions)(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=pi)
    return model


def build_value_model(shape: Tuple[int]) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=shape, dtype=tf.float64)
    v = tf.keras.layers.Dense(1)(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=v)
    return model



if __name__ == '__main__':
    main()




