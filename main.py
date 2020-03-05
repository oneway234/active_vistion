from environmemts.env_acd import Active_vision_env
from learn.dqn_learn import DQN

MEMORY_CAPACITY = 2000
def run_acd():
    dqn = DQN()
    steps = 0
    for episode in range(1000):
        # initial observation
        print("initial observation")
        train_set, img, thing_label, diff = env.reset(episode) # observation:
        while True:
            # RL choose action based on observation
            action = dqn.choose_action(img)

            # RL take action and get next observation and reward
            reward, next_img, next_diff = env.step(train_set, img, thing_label, diff, action)

            s = img
            s_ = next_img
            r = reward
            a = action
            dqn.store_transition(s, a, r, s_)

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()  # 记忆库满了就进行学习

            if stopping_criterion(next_diff, steps):
                break
            steps += 1
            img = next_img


def stopping_criterion(next_diff, steps):
    if next_diff == 1 or steps >= 100:
        return True


if __name__ == '__main__':
    env = Active_vision_env()
    run_acd()
