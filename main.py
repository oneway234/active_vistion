from torch.optim import rmsprop
from agents.dqn import DQN
from learn.learn import OptimizerSpec, dqn_learing
from utils.schedules import LinearSchedule
from environmemts.env_acd import Active_vision_env
import torch

import numpy as np
import torch.nn as nn

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

def run_acd():
    steps = 0
    for episode in range(1000):
        # initial observation
        train_set, img, thing_label, diff = env.reset(episode) # observation:
        while True:
            # RL choose action based on observation
            action = "RL.choose_action(img, thing_label, diff)"

            # RL take action and get next observation and reward
            reward, next_img, next_diff = env.step(train_set, img, thing_label, diff, action)

            if stopping_criterion(next_diff, steps):
                break
            steps += 1


def stopping_criterion(next_diff, steps):
    if next_diff == 1 or steps >= 100:
        return True


if __name__ == '__main__':
    env = Active_vision_env()
