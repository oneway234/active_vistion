from torch.optim import rmsprop
from agents.dqn import DQN
from learn.learn import OptimizerSpec, dqn_learing
from utils.schedules import LinearSchedule
from environmemts.env_acd import Active_vision_env

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
    # the following code are not sure how to write
    optimizer_spec = OptimizerSpec(
        constructor=rmsprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        # env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )


def stopping_criterion(next_diff, steps):
    if next_diff == 1 or steps >= 100:
        return True


if __name__ == '__main__':
    env = Active_vision_env()
