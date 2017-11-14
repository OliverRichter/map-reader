# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import os
from collections import deque
import pygame, sys
import time
from pygame.locals import *

from environment.environment import Environment
from model.model import MapReaderModel
from constants import *
from train.experience import ExperienceFrame

FRAME_SAVE_DIR = "/tmp/mapreader_frames"

BLUE  = (128, 128, 255)
RED   = (255, 100, 100)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class MovieWriter(object):
  def __init__(self, file_name, frame_size, fps):
    """
    frame_size is (w, h)
    """
    self._frame_size = frame_size
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    self.vout = cv2.VideoWriter(file_name,fourcc,fps,frame_size,True)

  def add_frame(self, frame):
    """
    frame shape is (h, w, 3), dtype is np.uint8
    """
    self.vout.write(frame)

  def close(self):
    self.vout.release() 
    self.vout = None


class ValueHistory(object):
  def __init__(self):
    self._values = deque(maxlen=100)
    self.add_value(0)

  def add_value(self, value):
    self._values.append(value)

  @property    
  def is_empty(self):
    return len(self._values) == 0

  @property
  def values(self):
    return self._values


class Display(object):
  def __init__(self, display_size,model):
    pygame.init()
    self.surface = pygame.display.set_mode(display_size, 0, 24)
    pygame.display.set_caption('MAPREADER')

    self.action_size = Environment.get_action_size()
    self.global_network = model
    self.environment = Environment.create_environment(*DISPLAY_LEVEL)
    self.font = pygame.font.SysFont(None, 20)
    self.value_history = ValueHistory()
    self.step_count = 0
    self.episode_reward = 0
    self.episode_intrinsic_reward = 0
    self.state = self.environment.last_state
    self.replan = True
    self.path = []
    self.maze_size = DISPLAY_LEVEL[0]//40*2+7

  def update(self, sess):
    self.surface.fill(BLACK)
    self.process(sess)
    pygame.display.update()

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def scale_image(self, image, scale):
    return image.repeat(scale, axis=0).repeat(scale, axis=1)

  def draw_text(self, str, left, top, color=WHITE):
    text = self.font.render(str, True, color, BLACK)
    text_rect = text.get_rect()
    text_rect.left = left    
    text_rect.top = top
    self.surface.blit(text, text_rect)  

  def draw_center_text(self, str, center_x, top):
    text = self.font.render(str, True, WHITE, BLACK)
    text_rect = text.get_rect()
    text_rect.centerx = center_x
    text_rect.top = top
    self.surface.blit(text, text_rect)

  def show_map(self, m, left, top, scale, rate, label, location=np.zeros(shape=(63, 63)),exact_location=np.asarray([False])):
    """
    Show map
    """
    location_ = self.scale_image(location, 2)
    data = np.clip(m*rate, 0.0, 1.0)
    if len(data[0][0])==1:
      data = np.stack([data[:,:,0] for _ in range(3)], axis=2)
    if exact_location.any():
      y,x = (exact_location*126/21).astype(np.intc)
      x = 6*self.maze_size - 1 - x
      self.path.append([x,y])
      for x,y in self.path[-1:]:
        data[x, y, 0] = data[x, y, 0] + 1
        data[x, y, 1] = data[x, y, 1] - 1
        data[x, y, 2] = data[x, y, 2] - 1
    data[:, :, 0] = data[:, :, 0] - 5 * location_
    data[:, :, 1] = data[:, :, 1] - 5*location_
    data[:, :, 2] = data[:, :, 2] + 5*location_
    data_ = np.clip(data * 255.0, 0.0, 255.0)
    data = data_.astype(np.uint8)
    data = self.scale_image(data, scale)
    image = pygame.image.frombuffer(data, (len(m)*scale,len(m)*scale), 'RGB')
    self.surface.blit(image, (left, top))
    self.draw_center_text(label, left + (len(m)*scale)/2, top + (len(m)*scale)+8)


  def show_pixels(self,m, left, top, scale, rate, label, local_map=False):
    """
    Show pixels
    """
    m_ = np.pad(self.scale_image(m, scale),((1,1),(1,1)),'constant',constant_values=1.0)
    data_ = np.clip(m_ * 255.0*rate, 0.0, 255.0)
    data = data_.astype(np.uint8)
    data = np.stack([data for _ in range(3)], axis=2)
    image = pygame.image.frombuffer(data, (len(m_[0]),len(m_)), 'RGB')
    self.surface.blit(image, (left, top))
    self.draw_center_text(label, left + (len(m_[0]))/2, top + (len(m_))+8)
    if local_map:
      center_x = left + (len(m_[0]))/2
      center_y = top + (len(m_))/2
      pygame.draw.line(self.surface, RED, (center_x-4, center_y), (center_x+4, center_y), 1)
      pygame.draw.line(self.surface, RED, (center_x, center_y-4), (center_x, center_y+4), 1)

  def show_angle(self,angle):
    """
    Show current angle
    """
    angle = angle*2*np.pi

    top = 8
    left = 380
    width = 100
    height = 100
    bottom = top + width
    right = left + height

    pygame.draw.line(self.surface, WHITE, (left, top), (left, bottom), 1)
    pygame.draw.line(self.surface, WHITE, (right, top), (right, bottom), 1)
    pygame.draw.line(self.surface, WHITE, (left, top), (right, top), 1)
    pygame.draw.line(self.surface, WHITE, (left, bottom), (right, bottom), 1)

    middle_x = (left+right)/2
    middle_y = (top+bottom)/2

    pygame.draw.line(self.surface, WHITE, (middle_x,middle_y), (middle_x + 50*np.cos(angle), middle_y - 50*np.sin(angle)), 1)
    self.draw_center_text("Direction Facing", middle_x, bottom+8)


  def show_policy(self, pi,a):
    """
    Show action probability.
    """
    start_x = 960

    y = 400

    action = ['Turn left','Turn right', 'Go left', 'Go right','Go forward','Go backward']

    for i in range(len(pi)):
      if i==a:
        color = RED
      else:
        color = WHITE
      width = pi[i] * 100
      pygame.draw.rect(self.surface, color, (start_x, y, width, 10))
      self.draw_text(action[i],start_x+110,y)
      y += 20
    self.draw_center_text("Policy", start_x+50, y)
  
  def show_image(self, state):
    """
    Show input image
    """
    state_ = state['view'][:,:,:3] * 255.0
    data = state_.astype(np.uint8)
    data = self.scale_image(data, 4)
    image = pygame.image.frombuffer(data, (84*4,84*4), 'RGB')
    self.surface.blit(image, (8, 8))
    self.draw_center_text("Visual Input", 8+2*84, 2*8+4*84)

  def show_value(self):
    if self.value_history.is_empty:
      return

    min_v = float("inf")
    max_v = float("-inf")

    values = self.value_history.values

    for v in values:
      min_v = min(min_v, v)
      max_v = max(max_v, v)

    top = 600
    left = 1000
    width = 100
    height = 100
    bottom = top + width
    right = left + height

    d = max_v - min_v
    if d == 0:
      d = 1 # prevent division through 0
    last_r = 0.0
    for i,v in enumerate(values):
      r = (v - min_v) / d
      if i > 0:
        x0 = i-1 + left
        x1 = i   + left
        y0 = bottom - last_r * height
        y1 = bottom - r * height
        pygame.draw.line(self.surface, BLUE, (x0, y0), (x1, y1), 1)
      last_r = r

    pygame.draw.line(self.surface, WHITE, (left,  top),    (left,  bottom), 1)
    pygame.draw.line(self.surface, WHITE, (right, top),    (right, bottom), 1)
    pygame.draw.line(self.surface, WHITE, (left,  top),    (right, top),    1)
    pygame.draw.line(self.surface, WHITE, (left,  bottom), (right, bottom), 1)

    self.draw_center_text("Value Estimation", left + width / 2, bottom + 10)
    self.draw_center_text(str(values[-1]), left + width / 2, bottom + 30)

  def show_reward_prediction(self, rp_c, reward,left,top,label):
    start_x = left
    reward_index = 0
    if reward == 0:
      reward_index = 0
    elif reward > 0:
      reward_index = 1
    elif reward < 0:
      reward_index = 2

    y = top

    labels = ["0", "+", "-"]
    
    for i in range(len(rp_c)):
      width = rp_c[i] * 100

      if i == reward_index:
        color = RED
      else:
        color = WHITE
      pygame.draw.rect(self.surface, color, (start_x+15, y, width, 10))
      self.draw_text(labels[i], start_x, y-1, color)
      y += 20
    
    self.draw_center_text(label, start_x + 100/2, y)


  def process(self, sess):

    last_action_reward = ExperienceFrame.concat_action_and_reward(self.environment.last_action,
                                                                  self.action_size,
                                                                  self.environment.last_reward)
    map_input = self.environment.map

    pi_values, v_value, location,angle,value_map,reward_map,short_term_goal,angle_neurons, local_map_prediction, \
    local_map, actual_local_map, vlm_target,vlm_prediction,location_estimate,shift_weights = self.global_network.run_display_values(sess,
                                                                         self.environment.last_state,
                                                                         last_action_reward,
                                                                         map_input,
                                                                         self.replan)
    if self.replan:
      self.path = []
      self.step_count = 0
      self.episode_reward = 0
      self.episode_intrinsic_reward = 0
    self.replan = False
    self.value_history.add_value(v_value)
    action = self.choose_action(pi_values)
    state, reward, intrinsic_reward, terminal = self.environment.process(action, short_term_goal, shift_weights)
    self.replan = False
    if terminal:
        print('Steps needed: ', self.step_count)
        sys.stdout.flush()
        self.environment.reset(DISPLAY_LEVEL[0],np.random.randint(LEVEL_SET_SIZE))
        self.global_network.reset_state()
        self.replan = True
    self.episode_reward += reward
    self.episode_intrinsic_reward += intrinsic_reward
    self.step_count += 1


    self.show_image(self.state)
    self.show_angle(angle)
    self.show_pixels(np.reshape(angle_neurons,[1,30]),370, 176, 4, 1, "Discretized Angle")

    self.show_pixels(np.reshape(shift_weights,[3,3]),400, 250, 20, 1, "Egomotion Estimation")

    self.show_pixels(vlm_target,550, 8, 5, 1, "Visible Local Map Target",True)
    self.show_pixels(vlm_prediction,550, 176, 5, 1, "Visible Local Map Estimation",True)

    self.show_pixels(actual_local_map,725, 8, 5, 1, "Local Map Target",True)
    self.show_pixels(local_map_prediction,725, 176, 5, 1, "Local Map Estimation",True)

    self.show_pixels(local_map,900, 8, 5, 1, "Map Feedback Local Map",True)

    self.draw_text("Estimated Position: " + str(np.around(location_estimate)), 900, 220)
    self.draw_text("Actual Position:      " + str(np.asarray(self.state['position'][2], 'float')), 900, 240)
    self.draw_text("STEPS: {}".format(int(self.step_count)), 900, 260)
    self.draw_text("REWARD: {}".format(float(self.episode_reward)), 900, 280)
    self.draw_text("INTRINSIC REWARD: {}".format(float(self.episode_intrinsic_reward)), 900, 300)


    disp_map = np.reshape(map_input, [126, 126,1])
    self.show_map(disp_map,8,400,3,1,"Map",location,self.state['position'][1])

    self.show_map(self.scale_image(reward_map, 2), 400, 400, 3, 1, "Reward Map, R = 0, G = +, B = -")

    stg = np.asarray([[0, short_term_goal[2], 0],
                      [short_term_goal[3], short_term_goal[4], short_term_goal[1]],
                      [0, short_term_goal[0], 0]])
    self.show_pixels(stg, 840, 400, 20, 1, "Short Term")
    self.draw_center_text("Target Direction", 870, 490)

    rp_c = self.global_network.run_map_rp_c(sess, self.state, state, map_input)
    self.show_reward_prediction(rp_c, reward, 820, 600, "Reward Prediction")

    self.show_policy(pi_values,action)
    self.show_value()

    self.state = state
    time.sleep(DISPLAY_SLOW_DOWN)

  def get_frame(self):
    data = self.surface.get_buffer().raw
    return data
  
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
model = MapReaderModel(Environment.get_action_size(), -1, "/cpu:0", for_display=True)
model.prepare_loss()
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  sys.stdout.flush()
else:
  print("Could not find old checkpoint")
  exit()
display_size = (1200, 900)

display = Display(display_size,model)




clock = pygame.time.Clock()

running = True
recording = True
frame_saving = False
fps = 15

if recording:
  if not os.path.exists('/tmp/mapreader_movies/'):
    os.mkdir('/tmp/mapreader_movies/')
  writer = MovieWriter('/tmp/mapreader_movies/out.avi', display_size, fps)


if frame_saving:  
  frame_count = 0
  if not os.path.exists(FRAME_SAVE_DIR):
    os.mkdir(FRAME_SAVE_DIR)
    
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
      
  display.update(sess)
  clock.tick(fps)
  
  if recording or frame_saving:
    frame_str = display.get_frame()
    d = np.fromstring(frame_str, dtype=np.uint8)
    d = d.reshape((display_size[1], display_size[0], 3))
    if recording:
      writer.add_frame(d)
    else:
      frame_file_path = "{0}/{1:06d}.png".format(FRAME_SAVE_DIR, frame_count)
      cv2.imwrite(frame_file_path, d)
      frame_count += 1

if recording:
  writer.close()

pygame.display.quit()