import gym
import torch
import numpy as np
from environment.board import Sudoku
from config import LEFT_TIMES, MAX_SIZE, MAX_SCORE
import os



class SudokuEnv(gym.Env):

    def __init__(self, env_config):
        self.env = Sudoku(env_config["device"] if env_config else {})
        self.left_times = LEFT_TIMES
        self.state = torch.Tensor()
        self.x = 0
        self.y = 0
        # self.action_space = gym.spaces.Dict({
        #     "x": gym.spaces.Discrete(9),
        #     "y": gym.spaces.Discrete(9),
        #     "v": gym.spaces.Discrete(9)
        # })
        self.action_space = gym.spaces.Discrete(4+MAX_SIZE)
        # self.observation_space = gym.spaces.Box(
        #     # np.array(self.env.reset()),
        #     shape=(9, 9, 1), high=9, low=1)

    def step(self, action):
        self.left_times -= 1

        truncated = self.left_times <= 0
        done = False
        reward = -0.01

        if action == 0:
            self.x = (self.x+1) % MAX_SIZE
        elif action == 1:
            self.x = (self.x-1) % MAX_SIZE
        elif action == 2:
            self.y = (self.y+1) % MAX_SIZE
        elif action == 3:
            self.y = (self.y-1) % MAX_SIZE
        else:
            changed = self.env.updateBoard({
                "x": self.x, "y": self.y, "v": action
            })
            score = self.env.calcScore()
            reward = score/MAX_SCORE if score else -0.01
            if not changed:
                reward = -self.left_times
                truncated = True
            else:
                self.state[0] = self.state[1]
                self.state[1] = self.state[2]
                self.state[2] = self.env.board
                # self.state[2][self.x][self.y][0] += 10

        if reward == 1:
            reward = self.left_times
            done = True

        return self.state, reward, truncated, done, {}

    def reset(self, type=None):

        self.left_times = LEFT_TIMES
        self.x = 0
        self.y = 0
        self.state = self.env.reset(type=type).expand(3, -1, -1, -1)

        return self.state
