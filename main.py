import gym

if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    print(env.reset())


    for _ in range(10000):
        env.render()
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()

    env.close()