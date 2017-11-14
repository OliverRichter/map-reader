import numpy as np
# -*- coding: utf-8 -*-

LOCAL_T_MAX = 20 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = '/tmp/mapreader_checkpoints'
LOG_FILE = '/tmp/mapreader_log/'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 5e-3   # log_uniform high limit for learning rate
PARALLEL_SIZE = 16 # parallel thread size
NUMBER_OF_GPUS = 1

ENV_TYPE = 'lab' # 'lab' or 'gym' or 'maze'
LEVEL_SET_SIZE = 20 # HAS TO BE CHANGED IN LUA SCRIPT AS WELL!!! Number of levels to sample from per episode

INITIAL_ALPHA_LOG_RATE = 0.5 # log_uniform interpolate rate for learning rate
GAMMA = 0.99 # discount factor for rewards
PLANNING_DISCOUNT_FACTOR = 0.99
NUM_PLANNING_STEPS = 200 # Number of value iteration steps during planning
TEMPERATURE = 0.001 # short term goal sharpening factor
ENTROPY_BETA = 0.1 # entropy regurarlization constant
PIXEL_CHANGE_LAMBDA = 0.05 # 0.01 ~ 0.1 for Lab, 0.0001 ~ 0.01 for Gym
EXPERIENCE_HISTORY_SIZE = 2000 # Experience replay buffer size
LOCAL_MAP_WIDTH = 23
LOCAL_MAP_SIZE = LOCAL_MAP_WIDTH**2

USE_LOCATION_CROSSENTROPY_LOSS  = True
USE_LOCAL_MAP_LOSS              = True
USE_LOCATION_COORDINATE_LOSS    = True

MAX_TIME_STEP = 10 * 10**9
SAVE_INTERVAL_STEP = 10**5

GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = False # To use GPU, set True

DISPLAY_LEVEL = [7,np.random.randint(LEVEL_SET_SIZE)]
DISPLAY_SLOW_DOWN = 0 # sleep time in seconds between frames

