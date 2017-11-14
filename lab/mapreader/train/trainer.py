# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import sys
from collections import deque

from environment.environment import Environment
from model.model import MapReaderModel
from train.experience import Experience, ExperienceFrame
from constants import *

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


class Trainer(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    self.action_size = Environment.get_action_size()
    self.local_network = MapReaderModel(self.action_size, thread_index, device)
    self.local_network.prepare_loss()

    self.apply_gradients = grad_applier.minimize_local(self.local_network.total_loss,
                                                       global_network.get_vars(),
                                                       self.local_network.get_vars())
    
    self.sync = self.local_network.sync_from(global_network)
    self.experience = Experience(EXPERIENCE_HISTORY_SIZE)
    self.local_t = 0
    self.initial_learning_rate = initial_learning_rate
    self.episode_reward = 0
    self.maze_size = 5
    if self.thread_index in range(2):
      self.maze_size = 13
    elif self.thread_index in [2,3]:
      self.maze_size = 11
    elif self.thread_index in [4,5]:
      self.maze_size = 9
    elif self.thread_index in [6,7]:
      self.maze_size = 7
    self.level_seed = np.random.randint(LEVEL_SET_SIZE)
    # For log output
    self.prev_local_t = 0
    self.last_terminal_local_t = 0
    self.steps_buffer = deque()
    self.correct_exits = 0
    self.running = True

  def prepare(self):
    if self.running:
      self.environment = Environment.create_environment(self.maze_size,self.level_seed)
      print('Started trainer ',self.thread_index)
      self.apply_next_location_loss = 0.0
      sys.stdout.flush()

  def stop(self):
    self.environment.stop()
    self.last_terminal_local_t = self.local_t

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate
  
  @staticmethod
  def choose_action(pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)
    
  def set_start_time(self, start_time):
    self.start_time = start_time

  def _fill_experience(self, sess):
    """
    Fill experience buffer until buffer is full.
    """
    prev_state = self.environment.last_state
    last_action = self.environment.last_action
    last_reward = self.environment.last_reward
    last_intrinsic_reward = self.environment.last_intrinsic_reward
    last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                  self.action_size,
                                                                  last_reward)
    input_map = self.environment.map
    prev_localization_state, pi_, _, short_term_goal,shift_weights, location_distribution = self.local_network.run_policy_and_value(sess,
                                                                                               prev_state,
                                                                                               last_action_reward,
                                                                                               input_map,
                                                                                               replan=False)
    action = self.choose_action(pi_)

    new_state, reward, intrinsic_reward, terminal = self.environment.process(action, short_term_goal,shift_weights)

    frame = ExperienceFrame(prev_state, input_map, prev_localization_state, location_distribution, reward,
                            intrinsic_reward, action, terminal, last_action, last_reward, last_intrinsic_reward)
    self.experience.add_frame(frame)

    if terminal:
      self.level_seed = np.random.randint(LEVEL_SET_SIZE)
      self.environment.reset(self.maze_size,self.level_seed)
    if self.experience.is_full():
      print("Replay buffer filled--------------------------------------------------------------------------------------")
      sys.stdout.flush()

  def _process_base(self, sess, global_t,map_input):
    # [Base A3C]
    states = []
    actions = []
    batch_last_action_rewards = []
    rewards = []
    values = []

    terminal_end = False
    replan = (self.apply_next_location_loss == 0.0)

    start_localization_state = self.local_network.localization_state_out

    # t_max times loop
    for _ in range(LOCAL_T_MAX):
      self.local_t += 1

      # Previous state
      prev_state = self.environment.last_state
      last_action = self.environment.last_action
      last_reward = self.environment.last_reward
      last_intrinsic_reward = self.environment.last_intrinsic_reward
      last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                    self.action_size,
                                                                    last_reward)

      prev_localization_state, pi_, value_, short_term_goal,shift_weights,location_distribution = self.local_network.run_policy_and_value(sess,
                                                                                                      prev_state,
                                                                                                      last_action_reward,
                                                                                                      map_input,
                                                                                                      replan)
      replan = False

      action = self.choose_action(pi_)

      states.append(prev_state)
      actions.append(ExperienceFrame.get_action_neurons(action,self.action_size))
      batch_last_action_rewards.append(last_action_reward)
      values.append(value_)

      if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
        print("pi={}".format(pi_))
        print(" V={}".format(value_))

      # Process game
      new_state, reward, intrinsic_reward, terminal = self.environment.process(action,short_term_goal,shift_weights)

      frame = ExperienceFrame(prev_state, map_input, prev_localization_state, location_distribution, reward,
                              intrinsic_reward, action, terminal, last_action, last_reward, last_intrinsic_reward)

      # Store to experience
      self.experience.add_frame(frame)

      self.episode_reward += reward+intrinsic_reward

      rewards.append(reward + intrinsic_reward)

      if terminal:
        terminal_end = True
        if reward > 0: self.correct_exits += 1
        steps_needed = self.local_t - self.last_terminal_local_t
        self.last_terminal_local_t = self.local_t
        self.steps_buffer.append(steps_needed)
        if len(self.steps_buffer)>50:
            self.steps_buffer.popleft()
        print("Steps needed: ", steps_needed)
        print("score={}".format(self.episode_reward))
        self.episode_reward = 0

        if (np.mean(self.steps_buffer) < 100 + (self.maze_size-7)*20 and len(self.steps_buffer)==50):
          self.maze_size += 2
          if self.maze_size > 13:
            print(">>>>>>>>>>> REACHED END <<<<<<<<<<<")
            self.environment.stop()
            sys.stdout.flush()
            self.running = False
            break
          print(">>>>>> SWITCHING TO MAZES OF SIZE ", self.maze_size,"x",self.maze_size, " AT GLOBAL T ", global_t," <<<<<<<<<<<<<<<")
          sys.stdout.flush()
          #reset moving average
          self.correct_exits = 0
          self.steps_buffer = deque()

        self.level_seed = np.random.randint(LEVEL_SET_SIZE)
        self.environment.reset(self.maze_size,self.level_seed)
        self.local_network.reset_state()
        break

    last_action_reward = ExperienceFrame.concat_action_and_reward(action,self.action_size,reward)
    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, new_state, last_action_reward,frame.map)
      self.apply_next_location_loss = 1.0
    else:
      self.apply_next_location_loss = 0.0

    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_adv = []
    batch_R = []

    for(ri, si, Vi) in zip(rewards, states, values):
      R = ri + GAMMA * R
      adv = R - Vi

      batch_si.append(si)
      batch_adv.append(adv)
      batch_R.append(R)

    batch_si.reverse()
    batch_adv.reverse()
    batch_R.reverse()

    return batch_si, batch_last_action_rewards, actions, batch_adv, batch_R, start_localization_state

  def process(self, sess, global_t,summary_writer):
    # Fill experience replay buffer
    if not self.experience.is_full():
      self._fill_experience(sess)
      return 0

    start_local_t = self.local_t

    cur_learning_rate = self._anneal_learning_rate(global_t)

    apply_location_loss = self.apply_next_location_loss

    # Copy weights from shared to local
    sess.run( self.sync )

    # [Base]
    map_input = self.environment.map
    batch_si, batch_last_action_rewards, batch_actions, batch_adv, batch_R, start_localization_state = \
          self._process_base(sess,global_t, map_input)

    vlm_frames = np.random.choice(self.experience.get_frames(), LOCAL_T_MAX)

    feed_dict = {
      self.local_network.input: map(lambda st: st['view'], batch_si),
      self.local_network.map: [map_input],
      self.local_network.replan: True,  # force replanning with updated reward model
      self.local_network.old_plan: np.zeros([1, 63, 63, 4]),
      self.local_network.last_action_reward: batch_last_action_rewards,
      self.local_network.angle_neurons: map(lambda st: st['angle'][0], batch_si),
      self.local_network.initial_localization_state: start_localization_state,
      # loss inputs
      self.local_network.a: batch_actions,
      self.local_network.adv: batch_adv,
      self.local_network.r: batch_R,
      self.local_network.location_loss_gate: apply_location_loss,
      self.local_network.position_indices: map(lambda st: st['position'][2], batch_si),
      self.local_network.location_probability_target: map(lambda st: st['position'][0], batch_si),
      # visual local map network
      self.local_network.visual_local_map_target: map(lambda f: f.state['vlm'], vlm_frames),
      self.local_network.vlm_view_input: map(lambda f: f.state['view'], vlm_frames),
      self.local_network.vlm_angle: map(lambda f: f.state['angle'][0], vlm_frames),
      # [common]
      self.learning_rate_input: cur_learning_rate
    }

    # Map reward prediction
    map_rp_frames = []
    map_rp_classes = []
    for _ in range(LOCAL_T_MAX):
        rp_frame = self.experience.sample_rp_frame()
        rp_c = [0.0, 0.0, 0.0]
        if rp_frame.reward == 0:
            rp_c[0] = 1.0  # zero
        elif rp_frame.reward > 0:
            rp_c[1] = 1.0  # positive
        else:
            rp_c[2] = 1.0  # negative
        map_rp_frames.append(rp_frame)
        map_rp_classes.append(rp_c)
    feed_dict.update({self.local_network.map_rp_location_distribution:
                        map(lambda f: f.location_distribution,map_rp_frames),
                      self.local_network.map_rp_maps: map(lambda f: f.map,map_rp_frames),
                      # loss input
                      self.local_network.map_rp_c_target: map_rp_classes,
                      self.local_network.map_rp_loss_gate: 1.0})


    # Calculate gradients and copy them to global netowrk.
    sess.run( self.apply_gradients, feed_dict=feed_dict )
    self._print_log(global_t,sess,feed_dict,summary_writer)
    # Return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t

  def _print_log(self, global_t,sess,feed_dict,summary_writer):
    if (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      if (self.thread_index == 0):
        elapsed_time = time.time() - self.start_time
        steps_per_sec = global_t / elapsed_time
        print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
          global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
      if self.steps_buffer:
        print('--- Thread ',self.thread_index,' at global_t ', global_t,' reached ', self.correct_exits, ' exits in ',
              np.mean(self.steps_buffer), ' steps on average in mazes of size ', self.maze_size, 'x',self.maze_size)
        feed_dict.update({self.local_network.episode: self.maze_size,
                          self.local_network.steps: self.steps_buffer})
        summary_str = sess.run(self.local_network.summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()
        sys.stdout.flush()
