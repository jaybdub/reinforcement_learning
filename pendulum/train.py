from gym.envs.classic_control import PendulumEnv


if __name__ == '__main__':
    env = PendulumEnv()
    print(env.action_space)
    print(env.observation_space)
