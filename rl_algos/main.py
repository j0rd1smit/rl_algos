import gym
from gym.spaces import Discrete, Box

if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    print(env.action_space)
    print(env.action_space.low)
    print(env.action_space.high)

    print(isinstance(env.action_space, Box))
    print(isinstance(env.action_space, Discrete))

    env = gym.make("MountainCarContinuous-v0")
    print(env.action_space)
    print(env.action_space.low)
    print(env.action_space.high)
    print(isinstance(env.action_space, Box))
    print(isinstance(env.action_space, Discrete))

    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(isinstance(env.action_space, Box))
    print(isinstance(env.action_space, Discrete))

    env = gym.make("Acrobot-v1")
    print(env.action_space)
    print(isinstance(env.action_space, Box))
    print(isinstance(env.action_space, Discrete))
