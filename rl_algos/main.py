import gym

if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    print(env.action_space)
