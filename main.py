from environmemts.env_acd import Active_vision_env
from learn.dqn_learn import DQN
import os
import random
import cv2
import numpy as np
import torch

MEMORY_CAPACITY = 10
def run_acd():
    dqn = DQN()

    for episode in range(1):
        # initial observation
        print("episode:", episode, "initial observation....")
        steps = 0
        train_set, img, thing_label, diff, curr_bbox = env.reset(episode) # observation:
        while True:
            # RL choose action based on observation
            inimg = os.path.join(train_set, 'jpg_rgb', img)#read curr image
            action = random.randint(0, 6)


            # action = dqn.choose_action(curr_s)  # choose action


            # RL take action and get next observation and reward
            # print("next stste")
            reward, next_img, next_diff, next_bbox = env.step(train_set, img, thing_label, diff, action)
            # print("steps:", steps, "action:", action, "curr_img", img, "next_img", next_img, "diff", diff, "next_diff", next_diff, "reward", reward)
            inextimg = os.path.join(train_set, 'jpg_rgb', next_img)  # read next img


            # s = curr_s
            # s_ = next_s
            # r = reward
            # a = action
            # print("storing...")
            # print(s)
            # dqn.store_transition(s, a, r, s_)
            # print("counter:", dqn.memory_counter)
            # print('next_diff:', next_diff, 'steps', steps)
            # if dqn.memory_counter > MEMORY_CAPACITY:
            #     print("start learning...")
            #     dqn.learn()  # 记忆库满了就进行学习

            if stopping_criterion(next_diff, steps):
                print("stop")
                break
            # print("steps =", steps)
            steps += 1
            img = next_img
            diff = next_diff
    print("save the net")
    # torch.save(dqn, 'dqn.pkl')

# def read_img(img):
#     inimg = cv2.imread(img)
#     inimg = np.transpose(inimg, (2, 0, 1))
#     inimg = torch.unsqueeze(torch.FloatTensor(inimg), 0)
#     return inimg

def stopping_criterion(next_diff, steps):
    if next_diff == 1 or steps >= MEMORY_CAPACITY + 100:
        return True


if __name__ == '__main__':
    env = Active_vision_env()
    cuda_gpu = torch.cuda.is_available()
    if (cuda_gpu):
        run_acd()