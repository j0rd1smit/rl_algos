
import gym

from rl_algos.TabularQLearning.TabularQLearningAgent import TabularQLearningAgent

if __name__ == '__main__':
    env = gym.make("Taxi-v2")
    print(env.action_space)
    print(env.observation_space)
    possible_actions = [a for a in range(env.action_space.n)]


    agent = TabularQLearningAgent(possible_actions)

    for eps in range(200):
        total_reward = 0.0
        s = env.reset()
        step = 0

        for t in range(400):
            # get agent to pick action given state s.
            a = agent.select_action(s)

            next_s, r, done, _ = env.step(a)

            # train (update) agent for state s
            agent.train(s, a, r, next_s)

            s = next_s
            total_reward += r
            step += 1
            if done:
                break


        print(f"[{eps}] returns={total_reward} steps={step} eps={agent._eps}")

    """
    returns = 0
    steps = 0
    d = False
    o = env.reset()
    env.render()
    while not d:

        a = agent.select_action(o)
        o_next, r, d, _ = env.step(a)
        agent.train(o, a, r, o_next)
        o = o_next

        steps += 1
        returns += r
        env.render()
    """