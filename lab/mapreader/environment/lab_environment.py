# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import numpy as np
from collections import deque
import deepmind_lab
from constants import *
import sys

from environment import environment

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2

def load_map(maze_size,seed):
  new_map_txt_file_path = 'assets/dmlab_level_data/map_' + str(maze_size) + '_' + str(seed) + '.txt'

  f = open(new_map_txt_file_path, 'r')
  upscale = 6
  image_size = 21*upscale
  #initalize white field
  bw_img_data = [[1.0 for _ in range(image_size)] for _ in range(image_size)]
  for i, line in enumerate(f):
    for j, char in enumerate(line):
      if char == '*':
        for k in range(upscale):
          for l in range(upscale):
            #make walls black
            bw_img_data[(i * upscale + k)][j * upscale + l] = 0
      if char == 'G':
        for k in range(upscale):
          # mark goal with x
          bw_img_data[(i * upscale + k)][j * upscale + k] = 0
          bw_img_data[((i + 1) * upscale - 1 - k)][j * upscale + k] = 0
  f.close()

  map_size = (i+1) * upscale
  for k in range(image_size):
    for j in range(image_size):
      if k>=map_size or j>=map_size:
        #set area outside the maze to grey
        bw_img_data[k][j]=0.5

  return np.asarray(bw_img_data).reshape([image_size,image_size,1])

def worker(conn):
  env = deepmind_lab.Lab(
    'load_random_maze',
    ['RGB_INTERLACED','POS','ANG'],
    config={
      'fps': str(60),
      'width': str(84),
      'height': str(84)
    })
  conn.send(0)

  while True:
    command, arg = conn.recv()
    if command == COMMAND_RESET:
      env.reset(arg[0],arg[1])
      obs = env.observations()['RGB_INTERLACED']
      pos = env.observations()['POS']
      ang = env.observations()['ANG']
      conn.send([obs,pos,ang])
    elif command == COMMAND_ACTION:
      reward = env.step(arg, num_steps=4)
      terminal = not env.is_running()
      if not terminal:
        obs = env.observations()['RGB_INTERLACED']
        pos = env.observations()['POS']
        ang = env.observations()['ANG']
      else:
        obs = 0
        pos = 0
        ang = 0
      conn.send([obs, pos, ang, reward, terminal])
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
    _action(-20,   0,  0,  0, 0, 0, 0), # look_left
    _action( 20,   0,  0,  0, 0, 0, 0), # look_right
    #_action(  0,  10,  0,  0, 0, 0, 0), # look_up
    #_action(  0, -10,  0,  0, 0, 0, 0), # look_down
    _action(  0,   0, -1,  0, 0, 0, 0), # strafe_left
    _action(  0,   0,  1,  0, 0, 0, 0), # strafe_right
    _action(  0,   0,  0,  1, 0, 0, 0), # forward
    _action(  0,   0,  0, -1, 0, 0, 0), # backward
    #_action(  0,   0,  0,  0, 1, 0, 0), # fire
    #_action(  0,   0,  0,  0, 0, 1, 0), # jump
    #_action(  0,   0,  0,  0, 0, 0, 1)  # crouch
  ]

  @staticmethod
  def get_action_size():
    return len(LabEnvironment.ACTION_LIST)

  @staticmethod
  def inverse_actions(action1,action2):
    return np.all(np.equal(LabEnvironment.ACTION_LIST[action1]+LabEnvironment.ACTION_LIST[action2],
                           np.zeros(len(LabEnvironment.ACTION_LIST[0]))))

  def __init__(self,maze_size,seed):
    environment.Environment.__init__(self)
    self.conn, child_conn = Pipe()
    self.proc = Process(target=worker, args=(child_conn,))
    self.proc.start()
    self.conn.recv()
    self.reset(maze_size,seed)

  def reset(self, maze_size, seed):
    self.conn.send([COMMAND_RESET, [maze_size,seed]])
    self.maze_size = maze_size
    obs, pos, ang = self.conn.recv()

    self.map = load_map(maze_size,seed)
    self.padded_map = np.pad(self._subsample(self.map-0.5,6),
                             [[LOCAL_MAP_WIDTH//6,LOCAL_MAP_WIDTH//6],[LOCAL_MAP_WIDTH//6,LOCAL_MAP_WIDTH//6]],
                             'constant')
    position = self._preprocess_position(pos)
    angle = self._preprocess_angle(ang)
    self.last_state = {'view': self._preprocess_frame(obs),
               'position': position,
               'angle': angle,
               'egomotion': np.zeros(9),
               'intended_egomotion': np.zeros(9),
               'vlm': self._get_visual_local_map(self.padded_map,position[2],angle[1])}
    self.last_action = 0
    self.last_reward = 0
    self.last_intrinsic_reward = 0
    self.previous_short_term_goal = np.ones(5)


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

  def _preprocess_position(self,pos):
    #exact_position[0] = how much to the right, exact_position[1] = how much to the top
    exact_position = ((pos[:2]/100)).astype(np.float32)

    #hit_wall = True if and only if the position is the closest the agent can get to a wall
    hit_wall = 1e-5>np.min([(exact_position-0.16125)%1.0,(-exact_position-0.16125)%1.0])

    #discretization to position indice
    pos = (pos[:2] * 3 / 100).astype(np.intc)
    pos[1] = 3 * self.maze_size - pos[1] - 1
    position_indices = np.asarray([pos[1], pos[0]])

    #one hot encoding for training
    position = np.zeros(shape=(63, 63))
    position[pos[1]][pos[0]] = 1.0

    return position, exact_position, position_indices,hit_wall

  def _preprocess_angle(self,ang):
    # 3-hot encoding
    angle_neurons = np.zeros(30)
    angle_neurons[int(ang[1] // 12)%30] = 1.0
    angle_neurons[int(ang[1] // 12 + 1)%30] = 1.0
    angle_neurons[int(ang[1] // 12-1)%30] = 1.0

    # converting to unit scale
    angle = ang[1] / 360.0
    if angle < 0:
      angle = angle + 1.0

    return angle_neurons, angle

  def _get_visual_local_map(self,padded_map,pos,ang):
    inner_pos = 2-pos%3
    pos = pos//3
    local_map = padded_map[pos[0]-1:pos[0] + LOCAL_MAP_WIDTH//3+1, pos[1]-1:pos[1] + LOCAL_MAP_WIDTH//3+1]
    visual_field = np.asarray([[False]*(LOCAL_MAP_WIDTH+6)]*(LOCAL_MAP_WIDTH+6))
    visible_local_map = np.zeros([LOCAL_MAP_WIDTH,LOCAL_MAP_WIDTH])
    visible = [True]
    marks = [True,True,True,True]
    ang = ang - 0.5
    if ang<0:
      ang = ang + 1.0
    # determine visual field
    for ring_radius in range(LOCAL_MAP_WIDTH//2+4):
      ring_size = 8*ring_radius
      if ring_size == 0: ring_size = 1
      for i in range(2*ring_radius+1):
        ring_position = int(ang*ring_size+i)%ring_size
        if ring_position <= ring_size // 4:
          x = LOCAL_MAP_WIDTH // 2 - ring_radius + ring_position
          y = LOCAL_MAP_WIDTH // 2 - ring_radius
        elif ring_position <= 2 * ring_size // 4:
          x = LOCAL_MAP_WIDTH // 2 + ring_radius
          y = LOCAL_MAP_WIDTH // 2 - ring_radius + (ring_position - ring_size // 4)
        elif ring_position <= 3 * ring_size // 4:
          x = LOCAL_MAP_WIDTH // 2 + ring_radius - (ring_position - 2 * ring_size // 4)
          y = LOCAL_MAP_WIDTH // 2 + ring_radius
        else:
          x = LOCAL_MAP_WIDTH // 2 - ring_radius
          y = LOCAL_MAP_WIDTH // 2 + ring_radius - (ring_position - 3 * ring_size // 4)
        visual_field[x+3][y+3] = True

    # get visible local map elements
    for ring_radius in range(LOCAL_MAP_WIDTH//3+2):
      ring_size = 4 * ring_radius
      if ring_size == 0: ring_size = 1
      for i in range(ring_size):
        if visible[i]:
          x,y = self.ring_pos_to_xy(i,ring_size)
          in_visual_field = np.any(visual_field[3*x + inner_pos[0]:3*x + inner_pos[0]+3,
                                                3*y + inner_pos[1]:3*y + inner_pos[1]+3])
          if x <= LOCAL_MAP_WIDTH//3+1 and x >= 0 and y<=LOCAL_MAP_WIDTH//3+1 and y>=0:
            value = local_map[x,y]
          else:
            value = 0
          if in_visual_field:
            for l in range(3):
              x_ = 3*(x-1) + inner_pos[0] + l
              if x_<LOCAL_MAP_WIDTH and x_>=0:
                for k in range(3):
                  y_ = 3*(y-1) + inner_pos[1] + k
                  if y_<LOCAL_MAP_WIDTH and y_>=0:
                    visible_local_map[x_][y_] = value
          if value < 0 or not in_visual_field:
            visible[i] = False
      visible_1 = np.concatenate([visible[:ring_radius + 1], visible[ring_radius:2 * ring_radius + 1],
                                  visible[2 * ring_radius:3 * ring_radius + 1], visible[3 * ring_radius:],
                                  visible[:min(ring_radius,1)]])
      visible_2 = np.concatenate(
        [visible[:min(ring_radius,1)], visible[:ring_radius + 1], visible[ring_radius:2 * ring_radius + 1],
         visible[2 * ring_radius:3 * ring_radius + 1], visible[3 * ring_radius:]])
      visible = np.logical_and(np.logical_or(visible_1, visible_2), marks)
      marks = np.logical_and(visible_1, visible_2)
      r = ring_radius + 1
      marks_1 = np.concatenate(
        [marks[:r + 1], marks[r:2 * r + 1], marks[2 * r:3 * r + 1], marks[3 * r:], [marks[0]]])
      marks_2 = np.concatenate(
        [[marks[-1]], marks[:r + 1], marks[r:2 * r + 1], marks[2 * r:3 * r + 1], marks[3 * r:]])
      marks = np.logical_or(marks_1, marks_2)
    return visible_local_map

  def ring_pos_to_xy(self,ring_position,ring_size):
    ring_radius = ring_size//4
    if ring_radius<1: return LOCAL_MAP_WIDTH//6+1,LOCAL_MAP_WIDTH//6+1
    if ring_position < ring_radius:
      x = LOCAL_MAP_WIDTH//6 + ring_position
      y = LOCAL_MAP_WIDTH//6 - ring_radius + ring_position
    elif ring_position < 2 * ring_radius:
      x = LOCAL_MAP_WIDTH//6 + ring_radius - ring_position%ring_radius
      y = LOCAL_MAP_WIDTH//6 + ring_position%ring_radius
    elif ring_position < 3 * ring_radius:
      x = LOCAL_MAP_WIDTH//6 - ring_position%ring_radius
      y = LOCAL_MAP_WIDTH//6 + ring_radius - ring_position%ring_radius
    else:
      x = LOCAL_MAP_WIDTH//6 - ring_radius + ring_position%ring_radius
      y = LOCAL_MAP_WIDTH//6 - ring_position%ring_radius
    return x+1,y+1

  def _get_intrinsic_reward(self,location_uncertainty,shift_weights):
      short_term_goal_vector = np.asarray([self.previous_short_term_goal[2]-self.previous_short_term_goal[0],
                                           self.previous_short_term_goal[3]-self.previous_short_term_goal[1]])
      egomotion_vector = np.asarray([np.sum(shift_weights[:3]-shift_weights[6:9]),
                                     shift_weights[0]+shift_weights[3]+shift_weights[6]
                                     - shift_weights[2]-shift_weights[5]-shift_weights[8]])
      return np.dot(short_term_goal_vector,egomotion_vector) + (self.previous_short_term_goal[4] - location_uncertainty)

  def process(self, action, short_term_goal, shift_weights):
    real_action = LabEnvironment.ACTION_LIST[action]
    self.conn.send([COMMAND_ACTION, real_action])
    obs, pos, ang, reward, terminal = self.conn.recv()

    if not terminal:
      position = self._preprocess_position(pos)
      angle = self._preprocess_angle(ang)
      state = {'view': self._preprocess_frame(obs),
               'position': position,
               'angle': angle,
               'vlm': self._get_visual_local_map(self.padded_map,position[2],angle[1])}

      if reward>0:
          terminal = True

      intrinsic_reward = self._get_intrinsic_reward(short_term_goal[4],shift_weights)
      # small negative reward for running into a wall
      if state['position'][3]:
        reward = reward - 0.1
    else:
      state = self.last_state
      intrinsic_reward = 0

    self.previous_short_term_goal = short_term_goal
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    self.last_intrinsic_reward = intrinsic_reward
    return state, reward, intrinsic_reward, terminal
