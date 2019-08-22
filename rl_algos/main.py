import gym

if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2")
    print(env.action_space)
    print(env.action_space.high)
    print(env.action_space.low)
    print(env.action_space.sample())
