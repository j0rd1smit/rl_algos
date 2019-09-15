import operator as op
from functools import reduce
from typing import Tuple, List

from tqdm import trange

from rl_algos.cross_entropy_table.CrossEntropyLogger import CrossEntropyLogger
from rl_algos.cross_entropy_table.CrossEntropyTable import CrossEntropyTable, Episode
from rl_algos.cross_entropy_table.ThieveryEnv import ThieveryEnv


def main() -> None:
    lr = 1e-8
    percentile = 50

    epochs = 1000
    n_sessions = 2500
    log_every = 100

    house_values = [10., 20., 30., 100., 200., 300., 1000., 2000., 3000., 10000.]
    #house_values = [1000., 500., 400., 300., 100.]

    max_n_guards = 3
    max_n_roberies = 3
    env = ThieveryEnv(house_values, max_n_guards, max_n_roberies)

    n_states_guard = env.n_states_guard
    n_action_guard = env.n_actions_guard
    agent_guard = CrossEntropyTable(n_states_guard, n_action_guard, lr, percentile)
    logger_guard = CrossEntropyLogger(percentile)

    n_states_thief = env.n_states_thief
    n_action_thief = env.n_actions_thief
    agent_thief = CrossEntropyTable(n_states_thief, n_action_thief, lr, percentile)
    logger_thief = CrossEntropyLogger(percentile)

    for i in trange(epochs):
        episodes_guard, episodes_thief = generate_session(env, agent_guard, agent_thief, n_sessions)

        agent_guard.update(episodes_guard)
        agent_thief.update(episodes_thief)

        logger_guard.log(episodes_guard)
        logger_thief.log(episodes_thief)

        if i % log_every == 0:
            print()
            print(f"[{i}] guard_min={logger_guard.mins[-1]} guard_max={logger_guard.maxs[-1]} guard_threshold={logger_guard.thresholds[-1]} guard_avg={logger_guard.avgs[-1]}")
            print(f"[{i}] thief_min={logger_thief.mins[-1]} thief_max={logger_thief.maxs[-1]} thief_threshold={logger_thief.thresholds[-1]} thief_avg={logger_thief.avgs[-1]}")

    logger_guard.plot()
    logger_thief.plot()




def generate_session(env: ThieveryEnv, guard: CrossEntropyTable, thief: CrossEntropyTable, n_sessions: int) -> Tuple[List[Episode], List[Episode]]:
    episodes_guard = []
    episodes_thief = []

    for _ in range(n_sessions):
        episode_guard, episode_thief = env.generate_episode(guard, thief)
        episodes_guard.append(episode_guard)
        episodes_thief.append(episode_thief)

    return episodes_guard, episodes_thief

def ncr(n: int, r: int) -> int:
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


if __name__ == '__main__':
    main()
