import os
import json
import random
from collections import Counter

def select_a_room(path): # Select the initial room's image set
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
    train_set = default_train_list[random.randint(0, len(default_train_list)-1)]
    test_set = default_test_list[random.randint(0, len(default_test_list)-1)]
    train_set = os.path.join(path, train_set)
    test_set = os.path.join(path, test_set)
    return train_set, test_set

def env_image_and_label(train_set, curr_img, label, action) :

    # set up curr_img and json paths
    images_path = os.path.join(train_set, 'jpg_rgb', curr_img)
    annotations_path = os.path.join(train_set, 'annotations.json')

    # load json data
    ann_file = open(annotations_path)
    annotations = json.load(ann_file)

    # set up for first image
    cur_image_name = images_path
    next_image_name = ''

    if action:
        # get the next image name to display based on the
        # user input, and the annotation.
        if action == 'w':
            next_image_name = annotations[curr_img]['forward']
        elif action == 'a':
            next_image_name = annotations[curr_img]['rotate_ccw']
        elif action == 's':
            next_image_name = annotations[curr_img]['backward']
        elif action == 'd':
            next_image_name = annotations[curr_img]['rotate_cw']
        elif action == 'e':
            next_image_name = annotations[curr_img]['left']
        elif action == 'r':
            next_image_name = annotations[curr_img]['right']
        elif action == 'h':
            next_image_name = curr_img
    reward = 0
    if next_image_name != '':
        if next_image_name != curr_img:
            cur_diff = count_reward(train_set, curr_img, label)
            next_diff = count_reward(train_set, next_image_name, label)

            if next_diff == -1:
                reward = -1
            elif cur_diff > next_diff and next_diff != -1:
                reward = 1
            elif cur_diff < next_diff and next_diff != -1:
                reward = -1

        elif next_image_name == cur_image_name:
            reward = 0

        curr_img = next_image_name

    return reward, curr_img

def count_reward(train_set, img, label):
    # set up curr_img and json paths
    annotations_path = os.path.join(train_set, 'annotations.json')

    # load json data
    ann_file = open(annotations_path)
    annotations = json.load(ann_file)

    # set up for first image
    diff = 0
    boxes = annotations[img]['bounding_boxes']
    if boxes:
        for i in boxes:
            if i[-2] == int(label[-2]):
                diff = i[-1]
            if diff == 0:
                diff = -1
    else:
        diff = -1 #

    return diff

def get_ini_img_label(train_set):
    """

    :param train_set: It means the path og the room
    :return: A random poistion with a image and label, this label is one of the top 3 mount labels in this room to
             make sure there is a process from diff 5 to diff 1.
    """
    path = os.path.join(train_set, 'annotations.json')

    total = 0
    with open(path) as f:
        fields = []
        for line in f.readlines():
            line = line.strip().split()
            line = line[0].strip(',').rstrip('{').rstrip('[').rstrip(':')
            fields.append(line)

        record = []
        for line_i in range(len(fields)):
            # print(fields[line_i])
            if (fields[line_i] == '"bounding_boxes"'):
                begin_i = line_i
            elif (fields[line_i] == ']'):
                final_i = line_i
                total += 1
                if (final_i - begin_i > 1):
                    for k in range(begin_i + 1, final_i):
                        if (fields[k][-2] == '4' or fields[k][-2] == '5'):
                            record.append(fields[k])
        label = []
        for s in record:
            label.append(s[1:-1].split(",")[-2])

        result = Counter(label)

        top3_label = {} # find out top3 num label
        for s in range(3):
            max_label = max(result, key=result.get)
            top3_label[max_label] = result[max_label]
            del result[max_label]
        the_label, _ = random.choice(list(top3_label.items()))

        record_img = {}
        for line_i in range(len(fields)):
            record = []
            if (fields[line_i] == '"bounding_boxes"'):
                begin_i = line_i
            elif (fields[line_i] == ']'):
                final_i = line_i
                total += 1
                if (final_i - begin_i > 1):
                    filename = fields[begin_i - 1]
                    for k in range(begin_i + 1, final_i):
                        if (fields[k][-2] == '4' or fields[k][-2] == '5'):
                            record.append(fields[k])
                    if record:
                        label_to_img = []
                        for s in record:
                            label_to_img.append(s[1:-1].split(","))
                        for s in label_to_img:
                            if s[-2] == the_label and s[-1] == '5':
                                record_img[filename] = s
        img, bbox = random.choice(list(record_img.items()))
        img = img[1:-1]
    return img, bbox

if __name__ == '__main__':
    path = "/home/wei/active vision/active_vistion_RL/dataset"
    train_set, test_set = select_a_room(path) # select a dataset from random room
    curr_img, bbox = get_ini_img_label(train_set) # get the initial image and bbox

    action = "w" # give a action "forward"

    # get the reward from current image and next image
    reward, curr_img = env_image_and_label(train_set, curr_img, bbox, action)


