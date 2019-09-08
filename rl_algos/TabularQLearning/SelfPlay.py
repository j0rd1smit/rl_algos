from rl_algos.envs.TicTacToEnvSelfPlay import TicTacToEnvSelfPlay
from rl_algos.TabularQLearning.Binarizer import Binarizer
from rl_algos.TabularQLearning.TabularQLearningAgent import TabularQLearningAgent

import matplotlib.pyplot as plt


if __name__ == '__main__':
    def binarize(state):
        res = tuple(state)
        return res

    env = Binarizer(TicTacToEnvSelfPlay(), binarize)
    """
    def possible_actions(state):
        actions = []
        for i in range(len(state)):
            if state[i] == 0:
                actions.append(i)
        #print(state)
        #print(actions)
        return actions
    """
    possible_actions = lambda s: list(range(9))


    agent = TabularQLearningAgent(possible_actions)

    def train_on_history(history):
        s, a, r, d = history[-1]
        #print(s, a, r, s, d)
        agent.train(s, a, r, s, d)
        for i in range(len(history) - 1):
            next_s, _, _, _  = history[len(history) -1 - i]
            s, a, r, d =  history[len(history) - 1 - i - 1]
            #print(s, a, r, next_s, d)
            agent.train(s, a, r, next_s, d, )
       # print()

    def total_reward_history(history):
        return sum(map(lambda x: x[2], history))

    draw, win_p1, win_p2 = 0, 0, 0
    log_rate = 50
    draw_pres, win_p1_pres, win_p2_pres = [], [], []


    for eps in range(10000):
        history_p1 = []
        history_p2 = []

        d = False
        s = env.reset()

        while not d:
            turn_p1 = env.player_one_turn
            a = agent.select_action(s)
            next_s, r, d, _ = env.step(a)

            if turn_p1:
                history_p1.append([s, a, r, d])
            else:
                history_p2.append([s, a, r, d])


            if d and turn_p1:
                s, a, _, _ = history_p2[-1]
                history_p2[-1] = [s, a, -r, d]
            elif d and not turn_p1:
                s, a, _, _ = history_p1[-1]
                history_p1[-1] = [s, a, -r, d]
                

            s = next_s

        if total_reward_history(history_p1) > 0:
            win_p1 += 1
        elif total_reward_history(history_p2) > 0:
            win_p2 += 1
        else:
            draw += 1

        train_on_history(history_p1)
        train_on_history(history_p2)
        #raise Exception("")
        if eps % log_rate == 0:
            print(f"[{eps}] Player 1 win_p1={win_p1 / log_rate} win_p2={win_p2 / log_rate} draw={draw / log_rate}")
            win_p1_pres.append(win_p1 / log_rate)
            win_p2_pres.append(win_p2 / log_rate)
            draw_pres.append(draw / log_rate)
            draw, win_p1, win_p2 = 0, 0, 0
            env.render()

    x = list(range(len(draw_pres)))
    plt.plot(x, win_p1_pres)
    plt.plot(x, win_p2_pres)
    plt.plot(x, draw_pres)
    plt.legend(['win_p1_pres', 'win_p2_pres', 'draw_pres'], loc='upper left')
    plt.show()
    #agent.render_q_table()