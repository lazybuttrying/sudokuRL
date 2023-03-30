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
        self.loc = torch.zeros(MAX_SIZE, MAX_SIZE, 1)
        self.action_space = gym.spaces.Discrete(4+MAX_SIZE)

    def step(self, action):
        self.left_times -= 1

        truncated = self.left_times <= 0
        done = False
        reward = -0.005
        self.loc[self.x][self.y][0] = 0

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
            reward = score/MAX_SCORE if score else -0.005
            if not changed:
                reward = -self.left_times
                truncated = True
            else:
                self.state[0] = self.state[1]
                self.state[1] = self.state[2]
                self.state[2] = self.env.board.clone()

        self.state[3] = self.state[4]
        self.state[4] = self.state[5]
        self.loc[self.x][self.y][0] = 1
        self.state[5] = self.loc.clone()

        if reward == 1.0:
            reward = self.left_times
            done = True

        return self.state, reward, truncated, done, {}

    def reset(self, type=None):
        self.left_times = LEFT_TIMES
        self.x = 0
        self.y = 0

        self.loc = torch.zeros(MAX_SIZE, MAX_SIZE, 1)
        self.loc[self.x][self.y][0] = 1
        self.state = self.env.reset(type=type).expand(6, -1, -1, -1).clone()

        self.state[3] = self.loc.clone()
        self.state[4] = self.loc.clone()
        self.state[5] = self.loc.clone()

        return self.state
