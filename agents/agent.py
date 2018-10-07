import numpy as np
from task import Task
from math import sqrt

class Random_Agent():
    def __init__(self, task, noise_scale = 0.1):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.episode = 0

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = noise_scale

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        self.episode += 1
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        temp = self.w * .5
        action = np.dot(state, self.w)  # simple linear policy
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        noise_scale = self.noise_scale / (3.1*float(self.episode))
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
        else:
            self.w = self.best_w
        self.w = self.w + noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        