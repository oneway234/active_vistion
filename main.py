from environmemts.env_acd import Active_vision_env
from learn.dqn_learn import DQN
import os
import cv2
import numpy as np
import torch

MEMORY_CAPACITY = 2000
def run_acd():
    dqn = DQN()
    steps = 0
    for episode in range(10):
        # initial observation
        print("initial observation")
        train_set, img, thing_label, diff = env.reset(episode) # observation:
        while True:
            # RL choose action based on observation
            inimg = os.path.join(train_set, 'jpg_rgb', img)#read curr img
            inimg = read_img(inimg)
            print("choose action...")
            action = dqn.choose_action(inimg) #choose action
            curr_s = dqn.eval_net.forward(inimg) #input img into net


            # RL take action and get next observation and reward
            print("next stste")
            reward, next_img, next_diff = env.step(train_set, img, thing_label, diff, action)

            inextimg = os.path.join(train_set, 'jpg_rgb', next_img)  # read next img
            inextimg = read_img(inextimg)
            next_s = dqn.eval_net.forward(inextimg)

            s = curr_s
            s_ = next_s
            r = reward
            a = action
            print("store")
            print("s", type(s), "\na:", a, "\nr:", r, "\ns:", s_)
            dqn.store_transition(s, a, r, s_)

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()  # 记忆库满了就进行学习

            if stopping_criterion(next_diff, steps):
                break
            steps += 1
            img = next_img

    torch.save(dqn, 'dqn.pkl')

def read_img(img):
    inimg = cv2.imread(img)
    inimg = np.transpose(inimg, (2, 0, 1))
    inimg = torch.unsqueeze(torch.FloatTensor(inimg), 0)
    return inimg

def stopping_criterion(next_diff, steps):
    if next_diff == 1 or steps >= 100:
        return True


if __name__ == '__main__':
    env = Active_vision_env()
    run_acd()
