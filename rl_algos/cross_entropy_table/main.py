from typing import List, Tuple

import gym
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from rl_algos.cross_entropy_table.CrossEntropyLogger import CrossEntropyLogger
from rl_algos.cross_entropy_table.CrossEntropyTable import Episode, CrossEntropyTable


def main() -> None:
    env = gym.make("Taxi-v2")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    lr = 0.05
    percentile = 55

    epochs = 20
    n_sessions = 250
    log_every = 10
    logger = CrossEntropyLogger(percentile)

    agent = CrossEntropyTable(n_states, n_actions, lr, percentile)

    for i in trange(epochs):
        episodes = [generate_episode(agent, env) for _ in range(n_sessions)]
        agent.update(episodes)

        logger.log(episodes)
        if i % log_every == 0:
            print(f"[{i}] min={logger.mins[-1]} max={logger.maxs[-1]} threshold={logger.thresholds[-1]} avg={logger.avgs[-1]}")


    logger.plot()



def generate_episode(agent: CrossEntropyTable, env: gym.Env, t_max: int = 1000) -> Episode:
    states, actions, rewards = [], [], []
    s = env.reset()

    for t in range(t_max):
        a = agent.get_action(s)
        s_next, r, done, _ = env.step(a)

        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = s_next
        if done:
            break

    return Episode(states, actions, rewards)


if __name__ == '__main__':
    main()