import gym
import time

env = gym.make('CartPole-v1', render_mode="human")  # V0 is not accepted now.
env.reset()


def init_gym():
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
        # env.reset()
    env.close()


def init_gym_2():
    count = 0
    for i in range(100):
        action = env.action_space.sample()  # random sample
        observation, reward, truncated, done, info = env.step(action)  # There are 5 returned parameters in 2023.
        if done:
            break
        env.render()
        count += 1
        print('Now it is the {} step'.format(count))


def init_gym_3():
    n_episode = 100
    env.reset()
    total_rewards = []
    for e in range(n_episode):
        total_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, truncated, done, info = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    # Finally, we calculate the expectation.
    print('Average total reward over {} episodes: {}'.format(n_episode, sum(total_rewards) / n_episode))


if __name__ == '__main__':
    # init_gym()
    # init_gym_2()
    init_gym_3()