import random

import gym


class ApproximateAgent:
    def __init__(self, observation):
        self.observation = observation
        self.weights = [0] * len(observation)
        self.exploration_rate = 0.8
        self.discount = 0.01
        self.alpha = 0.05

    def get_v(self, observation):
        return sum(map(lambda w, o: w * o, self.weights, observation))

    def step(self, env):
        if random.random() < self.exploration_rate:
            return env.action_space.sample()
        best_v = -1e10
        best_action = None
        for action in range(2):
            next_observation, reward, done, info = env.step(action)
            cur_v = self.get_v(next_observation)
            if best_v < cur_v:
                best_v = cur_v
                best_action = action
        return best_action

    def update(self, observation, next_observation, reward):
        for i in range(len(self.weights)):
            difference = reward + self.discount * self.get_v(next_observation) \
                         - self.get_v(observation)
            self.weights[i] += self.alpha * difference * observation[i]


def solve(problem_name):
    env = gym.make(problem_name)
    cur_observation = env.reset()
    agent = ApproximateAgent(cur_observation)
    for i_episode in range(200):
        agent.exploration_rate = 0.8 - i_episode / 100.0
        agent.alpha = max(0, agent.alpha - i_episode / 100.0)
        env.render()
        done = False
        cnt = 0
        while not done and cnt < 20:
            action = agent.step(env)
            next_observation, reward, done, info = env.step(action)
            agent.update(cur_observation, next_observation, reward)
            cur_observation = next_observation
            cnt += 1
        print(agent.weights)


if __name__ == '__main__':
    solve('CartPole-v0')
