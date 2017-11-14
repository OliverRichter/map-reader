# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import cv2
import sys
import os
import numpy as np
import deepmind_lab
import pyTGA

from environment import environment

# from constants import ENV_NAME

COMMAND_RESET = 0
COMMAND_ACTION = 1
COMMAND_TERMINATE = 2


def remove_map(map_name):
    prefix = '/tmp/dmlab_level_data/'
    txt_path = prefix + map_name + '.txt'
    os.remove(txt_path)
    prefix += 'baselab/'
    for sufix in ['.aas', '.bsp', '.map', '.pk3', '.srf']:
        os.remove(prefix + map_name + sufix)
    prefix += 'maps/'
    for sufix in ['.aas', '.bsp']:
        os.remove(prefix + map_name + sufix)


def replace_map(old_episode, old_seed, episode, seed):
    if old_episode is not None:
        remove_map('map_' + str(old_episode) + '_' + str(old_seed))
    new_map_txt_file_path = '/tmp/dmlab_level_data/map_' + str(episode) + '_' + str(seed) + '.txt'

    f = open(new_map_txt_file_path, 'r')

    image_size = 134
    bw_img_data = [[255 for _ in range(image_size)] for _ in range(image_size)]
    bw_img_data_flipped = [[255 for _ in range(image_size)] for _ in range(image_size)]
    for i, line in enumerate(f):
        upscale = int(image_size / (len(line) - 1))
        for j, char in enumerate(line):
            if char == '*':
                for k in range(upscale):
                    for l in range(upscale):
                        bw_img_data_flipped[image_size - 1 - (i * upscale + k)][j * upscale + l] = 0
                        bw_img_data[(i * upscale + k)][j * upscale + l] = 0
            if char == 'G':
                for k in range(upscale):
                    bw_img_data_flipped[image_size - 1 - (i * upscale + k)][j * upscale + k] = 0
                    bw_img_data_flipped[image_size - 1 - ((i + 1) * upscale - 1 - k)][j * upscale + k] = 0
                    bw_img_data[(i * upscale + k)][j * upscale + k] = 0
                    bw_img_data[((i + 1) * upscale - 1 - k)][j * upscale + k] = 0
            if char == 'P':
                for k in range(int(upscale / 2)):
                    bw_img_data_flipped[image_size - 1 - (i * upscale + int(upscale / 4) + k)][
                        j * upscale + int(upscale / 4)] = 0
                    bw_img_data_flipped[image_size - 1 - (i * upscale + int(upscale / 4) + k)][
                        j * upscale + int(3 * upscale / 4)] = 0
                    bw_img_data_flipped[image_size - 1 - (i * upscale + int(upscale / 4))][
                        j * upscale + int(upscale / 4) + k] = 0
                    bw_img_data_flipped[image_size - 1 - (i * upscale + int(3 * upscale / 4))][
                        j * upscale + int(upscale / 4) + k] = 0
                    bw_img_data[(i * upscale + int(upscale / 4) + k)][j * upscale + int(upscale / 4)] = 0
                    bw_img_data[(i * upscale + int(upscale / 4) + k)][j * upscale + int(3 * upscale / 4)] = 0
                    bw_img_data[(i * upscale + int(upscale / 4))][j * upscale + int(upscale / 4) + k] = 0
                    bw_img_data[(i * upscale + int(3 * upscale / 4))][j * upscale + int(upscale / 4) + k] = 0
    f.close()

    # image = pyTGA.Image(data=bw_img_data_flipped)
    # image.save('assets/textures/decal/lab_games/map')

    # image2 = pyTGA.Image(data=bw_img_data)
    # image2.save('assets/textures/decal/lab_games/map_not_flipped_' + str(episode) + '_' + str(seed))
    print('MAP LOADED')
    sys.stdout.flush()
    # print(bw_img_data)
    return np.asarray(bw_img_data).reshape([image_size, image_size, 1])


def worker(conn):
    env = deepmind_lab.Lab(
        'random_maze',
        ['RGB_INTERLACED'],
        config={
            'fps': str(60),
            'width': str(84),
            'height': str(84)
        })
    conn.send(0)

    while True:
        command, arg = conn.recv()
        if command == COMMAND_RESET:
            env.reset(arg[0], arg[1])
            obs = env.observations()['RGB_INTERLACED']
            conn.send(obs)
        elif command == COMMAND_ACTION:
            reward = env.step(arg, num_steps=4)
            terminal = not env.is_running()
            if not terminal:
                obs = env.observations()['RGB_INTERLACED']
            else:
                obs = 0
            conn.send([obs, reward, terminal])
        elif command == COMMAND_TERMINATE:
            break
        else:
            print("bad command: {}".format(command))
    env.close()
    conn.send(0)
    conn.close()


def _action(*entries):
    return np.array(entries, dtype=np.intc)


class LabEnvironment(environment.Environment):
    ACTION_LIST = [
        _action(-20, 0, 0, 0, 0, 0, 0),  # look_left
        _action(20, 0, 0, 0, 0, 0, 0),  # look_right
        # _action(  0,  10,  0,  0, 0, 0, 0), # look_up
        # _action(  0, -10,  0,  0, 0, 0, 0), # look_down
        _action(0, 0, -1, 0, 0, 0, 0),  # strafe_left
        _action(0, 0, 1, 0, 0, 0, 0),  # strafe_right
        _action(0, 0, 0, 1, 0, 0, 0),  # forward
        _action(0, 0, 0, -1, 0, 0, 0),  # backward
        # _action(  0,   0,  0,  0, 1, 0, 0), # fire
        # _action(  0,   0,  0,  0, 0, 1, 0), # jump
        # _action(  0,   0,  0,  0, 0, 0, 1)  # crouch
    ]

    @staticmethod
    def get_action_size():
        return len(LabEnvironment.ACTION_LIST)

    def __init__(self, episode, seed):
        environment.Environment.__init__(self)

        self.conn, child_conn = Pipe()
        self.proc = Process(target=worker, args=(child_conn,))
        self.proc.start()
        self.conn.recv()
        self.reset(None, None, episode, seed)

    def reset(self, old_episode, old_seed, episode, seed):
        self.conn.send([COMMAND_RESET, [episode, seed]])
        obs = self.conn.recv()

        self.map = self._preprocess_frame(replace_map(old_episode, old_seed, episode, seed))
        self.last_state = self._preprocess_frame(obs)
        self.last_action = 0
        self.last_reward = 0

    def stop(self):
        self.conn.send([COMMAND_TERMINATE, 0])
        ret = self.conn.recv()
        self.conn.close()
        self.proc.join()
        print("lab environment stopped")

    def _preprocess_frame(self, image):
        image = image.astype(np.float32)
        image = image / 255.0
        return image

    def process(self, action):
        real_action = LabEnvironment.ACTION_LIST[action]

        self.conn.send([COMMAND_ACTION, real_action])
        obs, reward, terminal = self.conn.recv()

        if not terminal:
            state = self._preprocess_frame(obs)
        else:
            state = self.last_state

        pixel_change = self._calc_pixel_change(state, self.last_state)
        self.last_state = state
        self.last_action = action
        self.last_reward = reward
        return state, reward, terminal, pixel_change
