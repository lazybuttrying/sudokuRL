import gym
import torch
import numpy as np
from environment.board import Sudoku
import os

LEFT_TIMES = 200


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
        self.action_space = gym.spaces.Discrete(13)
        # self.observation_space = gym.spaces.Box(
        #     # np.array(self.env.reset()),
        #     shape=(9, 9, 1), high=9, low=1)

    def step(self, action):
        self.left_times -= 1
        reward = -0.01

        if action == 0:
            self.x = (self.x+1) % 9
        elif action == 1:
            self.x = (self.x-1) % 9
        elif action == 2:
            self.y = (self.y+1) % 9
        elif action == 3:
            self.y = (self.y-1) % 9
        else:
            changed = self.env.updateBoard({
                "x": self.x, "y": self.y, "v": action
            })
            score = self.env.calcScore()
            reward = score/27.0 if score else -0.01

        self.state[0] = self.state[1]
        self.state[1] = self.state[2]
        self.state[2] = self.env.board
        # self.state = torch.stack([self.state, self.env.board], dim=0)[1:]

        score = self.env.calcScore()

        truncated = self.left_times <= 0
        # if not changed:
        #     reward = -LEFT_TIMES
        #     truncated = True

        done = reward == 1
        info = {"score": score}
        return self.state, reward, truncated, done, info

    def reset(self, type=None):
        self.left_times = LEFT_TIMES
        self.state = self.env.reset(type=type).expand(3, -1, -1, -1)

        return self.state
