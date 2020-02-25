from torch.optim import rmsprop
from agents.dqn import DQN
from learn.learn import OptimizerSpec, dqn_learing
from utils.schedules import LinearSchedule
import environmemts.env_acd as acv_env

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

default_train_list = [
                          'Home_002_1',
                          'Home_003_1',
                          'Home_003_2',
                          'Home_004_1',
                          'Home_004_2',
                          'Home_005_1',
                          'Home_005_2',
                          'Home_006_1',
                          'Home_014_1',
                          'Home_014_2',
                          'Office_001_1'

    ]
default_test_list = [
                          'Home_001_1',
                          'Home_001_2',
                          'Home_008_1'
    ]

def main(env, num_timesteps):

    def stopping_criterion(env):
        # when return label = 1 for 3 times
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=rmsprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
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

if __name__ == '__main__':
    path = "/home/wei/active vision/active_vistion_RL/dataset"

    # Get an initial poistion.
    train_set, test_set = acv_env.select_a_room(path)   # return dataset's path
    curr_img, bbox = acv_env.get_ini_img_label(train_set)  # Get a initial image and label.

    # Run traing in a label in diff = 5.
    action = "w"
    reward, curr_img = acv_env.env_image_and_label(train_set, curr_img, bbox, action)

    # main(env, task.max_timesteps)