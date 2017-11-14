# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from constants import *
from model.localization_rnn import localizationCell


# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer

def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


class MapReaderModel(object):
  """
  MapReader algorithm network model.
  """
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device,
               for_display=False):
    self._device = device
    self._action_size = action_size
    self._thread_index = thread_index
    self._create_network(for_display)

  def _create_network(self, for_display):
    scope_name = "net_{0}".format(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name):

      self._create_visual_input_network()
      self._create_vlm_network()

      self._create_localization_network()
      self._create_map_reading_network()
      self._create_map_rp_network()

      if for_display:
        self._create_map_reading_network_for_display()

      self.reset_state()

      self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)

# ----------------------------------------------------------------------------------------------------------------------

  def _create_visual_input_network(self):
      # State (Base image input)
      self.input = tf.placeholder("float", [None, 84, 84, 3],name='view')
      self.angle_neurons = tf.placeholder("float", [None, 30], name='angle_neurons')
      self.visual_features = self._visual_layers(self.input,self.angle_neurons)

  def _visual_layers(self,view_input,angle,reuse=False):
      conv_output = self._conv_layers(view_input, reuse)
      with tf.variable_scope("visual", reuse=reuse):
          # Weights
          W_fc_1, b_fc_1 = self._fc_variable([2592, LOCAL_MAP_SIZE], "fc_1")
          W_fc_2, b_fc_2 = self._fc_variable([LOCAL_MAP_SIZE+30, LOCAL_MAP_SIZE], "fc_2")
          W_fc_3, b_fc_3 = self._fc_variable([LOCAL_MAP_SIZE, 2*LOCAL_MAP_SIZE], "fc_3")

          output_fc_1 = tf.nn.relu(tf.matmul(conv_output, W_fc_1) + b_fc_1)
          output_fc_2 = tf.nn.relu(tf.matmul(tf.concat([output_fc_1, angle], 1), W_fc_2) + b_fc_2)
          output_fc_3 = tf.reshape(tf.clip_by_value(tf.matmul(output_fc_2, W_fc_3) + b_fc_3,-0.5,0.5),[-1,LOCAL_MAP_SIZE,2])
          visual_local_map = tf.multiply(output_fc_3[:,:,0],output_fc_3[:,:,1]+0.5)
          return visual_local_map

  def _conv_layers(self, state_input, reuse):
    with tf.variable_scope("conv", reuse=reuse):
      # Weights
      W_conv1, b_conv1 = self._conv_variable([8, 8, 3, 16],  "conv1")
      W_conv2, b_conv2 = self._conv_variable([4, 4, 16, 32], "conv2")


      # Nodes
      h_conv1 = tf.nn.relu(self._conv2d(state_input, W_conv1, 4) + b_conv1) # stride=4
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1,     W_conv2, 2) + b_conv2) # stride=2

      # Nodes
      conv_output_flat = tf.reshape(h_conv2, [-1, 2592])
      # (-1,9,9,32) -> (-1,2592)
      return conv_output_flat

# ----------------------------------------------------------------------------------------------------------------------

  def _create_localization_network(self):
    # Last action and reward
    self.last_action_reward = tf.placeholder("float", [None, self._action_size+1], name='last_action_reward')

    input_length = self._action_size + 31 + LOCAL_MAP_SIZE
    self.localization_cell = localizationCell(LOCAL_MAP_WIDTH, input_length)

    self.map = tf.placeholder("float", [1, 126, 126, 1], name='map')
    #resize map and shift to range [-0.5, 0.5]
    self.map_63 = tf.image.resize_area(self.map,[63,63]) - 0.5
    self.localization_cell.set_map(self.map_63)

    # Localization initial state
    self.initial_localization_state = tf.placeholder(tf.float32, [1, LOCAL_MAP_WIDTH+2,LOCAL_MAP_WIDTH+2,2]),tf.placeholder(tf.float32,[1,9])

    # localization layer
    localization_input = tf.concat([tf.stop_gradient(self.visual_features), self.last_action_reward, self.angle_neurons], 1)

    self.location_probability, self.localization_state, self.location_believe, self.location_uncertainty, self.shift_weights = \
        self._localization_layer(localization_input,input_length,self.initial_localization_state)


  def _localization_layer(self, localization_input, input_length, initial_state_input, reuse=False):
    with tf.variable_scope("localization", reuse=reuse) as scope:
        step_size = tf.shape(localization_input)[:1]
        input_reshaped = tf.reshape(localization_input, [1, -1, input_length])
        localization_output, localization_state = tf.nn.dynamic_rnn(self.localization_cell,
                                                     input_reshaped,
                                                     initial_state=initial_state_input,
                                                     sequence_length=step_size,
                                                     time_major=False,
                                                     scope=scope)
        location_believe = localization_output[:,:,:63**2]
        location_uncertainty = localization_output[0,:,63**2:63**2+1]
        shift_weights = localization_output[0,:,63**2+1:]

        location_probability = tf.reshape(tf.nn.softmax(location_believe), [-1, 63, 63,1])
        return location_probability, localization_state, tf.reshape(location_believe,[-1,63**2]),location_uncertainty,shift_weights

# ----------------------------------------------------------------------------------------------------------------------

  def _create_map_reading_network(self):
      # Map input
      self.replan = tf.placeholder("bool", [], name='replan')
      self.old_plan = tf.placeholder("float", [1, 63, 63, 4], name='old_plan')

      # Lazy planning
      self.plan = tf.cond(self.replan,
                               lambda: self._plan(self.map),
                               lambda: self.old_plan)
      plan_reshaped = tf.nn.softmax(tf.reshape(self.plan,[63*63,4])/TEMPERATURE)
      # Plan querry
      local_plan = tf.stop_gradient(tf.matmul(tf.reshape(self.location_probability,[-1,63*63]),plan_reshaped))
      self.short_term_goal = tf.concat([local_plan,
                                        tf.stop_gradient(self.location_uncertainty)],1)

      orientation = self._orientation_lstm_layers(tf.concat([self.short_term_goal,
                                                             self.angle_neurons,
                                                             self.last_action_reward[:,self._action_size:],
                                                             1.0-tf.reduce_max(local_plan,axis=1,keep_dims=True)
                                                             ], 1))
      # Policy and Value output
      self.pi = self._policy_layer(orientation)  # policy output
      self.v = self._value_layer(orientation)  # value output

  def _plan(self,map,reuse=False):
      reward_model = self._model_creator(map,reuse)
      plan = self._recursive_planning_module(reward_model,reuse)
      return plan

  def _model_creator(self,map_input,reuse=False):
    with tf.variable_scope("model", reuse=reuse) as scope:
      # Weights
      W_conv1, b_conv1 = self._conv_variable([6, 6, 1, 4], "model_conv1")
      W_conv2, b_conv2 = self._conv_variable([3, 3, 4, 3], "model_conv2")

      # Nodes
      h_conv1 = tf.nn.relu(self._conv2d(map_input, W_conv1, 2, "SAME") + b_conv1)  # stride=3
      model = tf.nn.relu(self._conv2d(h_conv1, W_conv2, 1, "SAME") + b_conv2)  # stride=1
      return model

  def _recursive_planning_module(self, raw_model, reuse=False):
    with tf.variable_scope("plan", reuse=reuse):
        # Weights for moving north, east, south or west
        W_VtoQ = tf.convert_to_tensor([[[[0, 0, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 0, 0]]],
                                       [[[0, 0, 0, 1]], [[0, 0, 0, 0]], [[0, 1, 0, 0]]],
                                       [[[0, 0, 0, 0]], [[1, 0, 0, 0]], [[0, 0, 0, 0]]]],dtype="float")

        averaged_model = tf.nn.depthwise_conv2d(raw_model,tf.ones([3,3,3,9]),[1,3,3,1],"VALID")
        averaged_model = tf.transpose(tf.reshape(averaged_model,[-1,21,21,3,3,3]),[0,1,4,2,5,3])
        averaged_model = tf.reshape(averaged_model,[-1,63,63,3])
        planning_model = tf.nn.softmax(averaged_model/0.1)
        connectivity_model = tf.clip_by_value(planning_model[:,:,:,:1]-planning_model[:,:,:,2:] + 0.5,0,1)
        reward_model = tf.clip_by_value(planning_model[:,:,:,1:2] + PLANNING_DISCOUNT_FACTOR * connectivity_model,0,1)
        V = reward_model
        # NUM_PLANNING_STEPS value iteration steps
        for _ in range(NUM_PLANNING_STEPS):
            Q = self._conv2d(V, W_VtoQ, 1,"SAME") # stride = 1, padding = 'SAME'
            V = tf.reduce_max(tf.multiply(Q,reward_model), axis=3,keep_dims=True)
        return Q

  def _orientation_lstm_layers(self,fc_input,reuse=False):
      with tf.variable_scope("orientation", reuse=reuse) as scope:
          self.W_fc_1, self.b_fc_1 = self._fc_variable([37, 32], "fc_1")
          W_fc_2, b_fc_2 = self._fc_variable([32, 32], "fc_2")

          hidden_neurons = tf.nn.relu(tf.matmul(fc_input, self.W_fc_1) + self.b_fc_1)
          orientation = tf.nn.relu(tf.matmul(hidden_neurons, W_fc_2) + b_fc_2)
          return orientation

  def _policy_layer(self, features, reuse=False):
    with tf.variable_scope("policy", reuse=reuse) as scope:
      # Policy (output)# Weight for policy output layer
      W_fc_p, b_fc_p = self._fc_variable([32, self._action_size], "fc_p")
      self.pi_logits = tf.matmul(features, W_fc_p) + b_fc_p
      pi = tf.nn.softmax(self.pi_logits)
      return pi

  def _value_layer(self, features, reuse=False):
    with tf.variable_scope("value", reuse=reuse) as scope:
      # Weight for value output layer
      W_fc_v, b_fc_v = self._fc_variable([32, 1], "fc_v")
      v_ = tf.matmul(features, W_fc_v) + b_fc_v
      v = tf.reshape( v_, [-1] )
      return v

  def _create_map_reading_network_for_display(self):
      averaged_model = tf.nn.depthwise_conv2d(self._model_creator(self.map, reuse=True), tf.ones([3, 3, 3, 9]), [1, 3, 3, 1], "VALID")
      averaged_model = tf.transpose(tf.reshape(averaged_model, [-1, 21, 21, 3, 3, 3]), [0, 1, 4, 2, 5, 3])
      averaged_model = tf.reshape(averaged_model, [-1, 63, 63, 3])
      planning_model = tf.nn.softmax(averaged_model / 0.1)
      self.reward_model = tf.nn.softmax(self._model_creator(self.map, reuse=True)) #planning_model #

# ----------------------------------------------------------------------------------------------------------------------

  def _create_vlm_network(self):
      self.vlm_view_input = tf.placeholder('float', [None, 84, 84, 3])
      self.vlm_angle = tf.placeholder('float', [None, 30])
      self.visual_local_map = self._visual_layers(self.vlm_view_input, self.vlm_angle, reuse=True)

# ----------------------------------------------------------------------------------------------------------------------

  def _create_map_rp_network(self):
    self.map_rp_maps = tf.placeholder("float", [None,126,126,1],name='map_rp_maps')
    self.map_rp_location_distribution = tf.placeholder(tf.float32,[None, 63,63,1])

    blured_location = tf.nn.conv2d(self.map_rp_location_distribution,tf.ones([3,3,1,1])/9,[1,1,1,1],"SAME")
    rp_models = self._model_creator(self.map_rp_maps,reuse=True)
    rp_model_reshaped = tf.reshape(rp_models, [-1,63 * 63, 3])
    location_probability_distribution_ = tf.stop_gradient(tf.reshape(blured_location, [-1,1,63 * 63]))
    self.map_rp_c = tf.nn.softmax(tf.reshape(tf.matmul(location_probability_distribution_,rp_model_reshaped),[-1,3]))
    # (1,3)

# ----------------------------------------------------------------------------------------------------------------------
# Losses
  def _a3c_loss(self):
    # [base A3C]
    # Taken action (input for policy)
    self.a = tf.placeholder("float", [None, self._action_size],name='action')

    # Advantage (R-V) (input for policy)
    self.adv = tf.placeholder("float", [None],name='advantage')

    # Avoid NaN with clipping when value in pi becomes zero
    log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

    # Policy entropy
    entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

    # Policy loss (output)
    policy_loss = -tf.reduce_sum(tf.reduce_sum(tf.multiply(log_pi, self.a), reduction_indices=1) * self.adv + entropy * ENTROPY_BETA)

    # R (input for value target)
    self.r = tf.placeholder("float", [None],name='value')

    # Value loss (output)
    # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
    value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

    loss = policy_loss + value_loss
    return loss

  def _visual_local_map_loss(self):
      self.visual_local_map_target = tf.placeholder("float",[None,LOCAL_MAP_WIDTH,LOCAL_MAP_WIDTH])
      visual_local_map_target = tf.reshape(self.visual_local_map_target,[-1,LOCAL_MAP_SIZE])
      vlm_loss = tf.nn.l2_loss(visual_local_map_target-self.visual_local_map)
      return vlm_loss

  def _map_rp_loss(self):
      # reward prediction target. one hot vector
      self.map_rp_c_target = tf.placeholder("float", [None, 3])
      self.map_rp_loss_gate = tf.placeholder("float",[])
      rp_c = tf.clip_by_value(self.map_rp_c, 1e-20, 1.0)
      rp_loss = -tf.reduce_sum(self.map_rp_c_target * tf.log(rp_c))
      return rp_loss*self.map_rp_loss_gate

  def _location_crossentropy_loss(self):
      self.location_probability_target = tf.placeholder("float", [None, 63, 63], name='location_target')
      location_probability_target = tf.reshape(self.location_probability_target, [-1, 63 ** 2])
      location_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.location_believe,
                                                                             labels=location_probability_target))
      return location_loss*self.location_loss_gate

  def _location_coordinate_loss(self):
    location_coordinates = tf.cast(self.position_indices,'float')
    coordinate_location_loss = tf.nn.l2_loss(location_coordinates-self.estimated_location_centroid)
    return coordinate_location_loss*self.location_loss_gate

  def _local_map_mismatch_loss(self):
      local_map_target = tf.reshape(self.local_map_target, [LOCAL_MAP_SIZE])
      local_map_state, _ = self.localization_state
      local_map_estimate = tf.reshape(local_map_state[0, 1:LOCAL_MAP_WIDTH + 1, 1:LOCAL_MAP_WIDTH + 1, 1],
                                      [LOCAL_MAP_SIZE])
      local_map_mismatch_loss = tf.nn.l2_loss(local_map_target - local_map_estimate)
      return local_map_mismatch_loss*self.location_loss_gate

  def prepare_loss(self):
    with tf.device(self._device):
      self.a3c_loss = self._a3c_loss()
      loss = self.a3c_loss

      self.vlm_loss = self._visual_local_map_loss()
      loss += self.vlm_loss

      self.map_rp_loss = self._map_rp_loss()
      loss += self.map_rp_loss

      self.location_loss_gate = tf.placeholder("float", [])
      self.position_indices = tf.placeholder('int32', [None,2])
      start_x = self.position_indices[-1,0]
      start_y = self.position_indices[-1,1]
      self.local_map_target = tf.pad(self.map_63[0, :, :, 0],
                                     [[LOCAL_MAP_WIDTH // 2, LOCAL_MAP_WIDTH // 2],
                                      [LOCAL_MAP_WIDTH // 2, LOCAL_MAP_WIDTH // 2]])[start_x:start_x + LOCAL_MAP_WIDTH,
                                                                                     start_y:start_y + LOCAL_MAP_WIDTH]
      coordinate_matrix = []
      for i in range(63):
          coordinate_matrix.append([tf.cast([i, j], 'float') for j in range(63)])
      self.estimated_location_centroid = tf.matmul(tf.reshape(self.location_probability, [-1, 63 ** 2]),
                                                   tf.reshape(coordinate_matrix, [63 ** 2, 2]))

      self.location_crossentropy_loss = self._location_crossentropy_loss()
      self.location_coordinate_loss = self._location_coordinate_loss()
      self.local_map_loss = self._local_map_mismatch_loss()

      if USE_LOCATION_CROSSENTROPY_LOSS:
        loss += self.location_crossentropy_loss

      if USE_LOCATION_COORDINATE_LOSS:
        loss += self.location_coordinate_loss

      if USE_LOCAL_MAP_LOSS:
        loss += self.local_map_loss


      self.total_loss = loss
      self.episode = tf.placeholder('int32',[])
      self.steps = tf.placeholder('float',[None])
      self.summary_op = tf.summary.merge([
        tf.summary.scalar("Episode",self.episode),
        tf.summary.scalar("Average Steps Needed", tf.reduce_mean(self.steps)),
        tf.summary.histogram("Steps Histogram",self.steps),
        tf.summary.scalar("Total loss", self.total_loss),
        tf.summary.scalar("A3C loss", self.a3c_loss),
        tf.summary.scalar("Location crossentropy loss", self.location_crossentropy_loss),
        tf.summary.scalar("Location coordinate loss", self.location_coordinate_loss),
        tf.summary.scalar("Local map loss", self.local_map_loss),
        tf.summary.scalar("Map reward prediction loss", self.map_rp_loss),
        tf.summary.scalar("Visual local map loss", self.vlm_loss),
        tf.summary.histogram("Pi logits",self.pi_logits)])

  def reset_state(self):
    self.localization_state_out = np.zeros([1, LOCAL_MAP_WIDTH+2,LOCAL_MAP_WIDTH+2,2]),np.zeros([1,9])
    self.plan_out = np.zeros([1,63,63,4])

# ----------------------------------------------------------------------------------------------------------------------
# Run functions

  def get_feed_dict(self,state,last_action_reward,map_input,replan):
      feed_dict = {self.input: [state['view']],
                   self.map: [map_input],
                   self.replan: replan,
                   self.old_plan: self.plan_out,
                   self.last_action_reward: [last_action_reward],
                   self.angle_neurons: [state['angle'][0]],
                   self.initial_localization_state: self.localization_state_out}
      return feed_dict

  def run_policy_and_value(self, sess, s_t, last_action_reward, map_input,replan):
    # This run_policy_and_value() is used when forward propagating,
    # so the step size is 1.
    prev_localization_state = self.localization_state_out
    pi_out, v_out, short_term_goal, shift_weights, self.plan_out, self.localization_state_out, location_distribution = \
        sess.run([self.pi, self.v, self.short_term_goal,self.shift_weights,self.plan, self.localization_state, self.location_probability],
                             feed_dict = self.get_feed_dict(s_t,last_action_reward,map_input,replan))

    return (prev_localization_state, pi_out[0], v_out[0], short_term_goal[0],shift_weights[0],location_distribution[0])

  def run_display_values(self,sess,s_t,last_action_reward,map_input,replan):
      feed_dict = self.get_feed_dict(s_t,last_action_reward,map_input,replan)
      feed_dict.update({self.position_indices: [s_t['position'][2]],
                        self.visual_local_map_target: [s_t['vlm']]})

      pi_out, v_out, angle_neurons, self.map_out, self.plan_out, short_term_goal,reward_map,self.location_out,\
      self.localization_state_out, local_map_target, vlm, location_estimate,shift_weights= \
            sess.run([self.pi, self.v, self.angle_neurons,self.map,self.plan,
                      self.short_term_goal, self.reward_model, self.location_probability, self.localization_state, self.local_map_target,
                      self.visual_features,self.estimated_location_centroid,self.shift_weights],
                               feed_dict=feed_dict)
      value_map = np.reshape(np.max(self.plan_out,axis=3),[63,63,1])
      localization_map_state,_ = self.localization_state_out
      local_map_prediction = np.reshape(localization_map_state[0,:,:,1]+0.5,[LOCAL_MAP_WIDTH+2,LOCAL_MAP_WIDTH+2])
      local_map = np.reshape(localization_map_state[0, :, :, 0] + 0.5,
                             [LOCAL_MAP_WIDTH + 2, LOCAL_MAP_WIDTH + 2])
      actual_local_map = np.reshape(local_map_target + 0.5,
                             [LOCAL_MAP_WIDTH, LOCAL_MAP_WIDTH])
      vlm_prediction = np.reshape(vlm,[LOCAL_MAP_WIDTH, LOCAL_MAP_WIDTH])+0.5
      vlm_target = np.reshape(s_t['vlm'],[LOCAL_MAP_WIDTH, LOCAL_MAP_WIDTH])+0.5
      angle = s_t['angle'][1]

      return (pi_out[0], v_out[0], self.location_out[0,:,:,0], angle, value_map, reward_map[0], short_term_goal[0],
              angle_neurons, local_map_prediction, local_map, actual_local_map, vlm_target, vlm_prediction,
              location_estimate[0], shift_weights[0])

  def run_value(self, sess, s_t, last_action_reward, map_input):
    # This run_value() is used for calculating V for bootstrapping at the
    # end of LOCAL_T_MAX time step sequence.
    # When next sequence starts, V will be calculated again with the same state using updated network weights,
    # so we don't update the localization state here.
    v_out = sess.run(self.v, feed_dict=self.get_feed_dict(s_t,last_action_reward,map_input,replan=False))
    return v_out[0]

  def run_map_rp_c(self, sess, state,new_state,map_input):
    # For display tool
    rp_c_out = sess.run(self.map_rp_c,
                        feed_dict={self.map_rp_location_distribution: self.location_out,
                                   self.map_rp_maps: [map_input]})
    return rp_c_out[0]
  
  def get_vars(self):
    return self.variables

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "MapReaderModel",[]) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  def _fc_variable(self, weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
    return weight, bias

  def _conv_variable(self, weight_shape, name, deconv=False, with_bias=True):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
      input_channels  = weight_shape[3]
      output_channels = weight_shape[2]
    else:
      input_channels  = weight_shape[2]
      output_channels = weight_shape[3]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape,
                             initializer=conv_initializer(w, h, input_channels))
    bias = None
    if with_bias:
      bias   = tf.get_variable(name_b, bias_shape,
                             initializer=conv_initializer(w, h, input_channels))
    return weight, bias

  def _conv2d(self, x, W, stride,padding="VALID"):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)

  def _get2d_deconv_output_size(self,
                                input_height, input_width,
                                filter_height, filter_width,
                                stride, padding_type):
    if padding_type == 'VALID':
      out_height = (input_height - 1) * stride + filter_height
      out_width  = (input_width  - 1) * stride + filter_width
      
    elif padding_type == 'SAME':
      out_height = input_height * row_stride
      out_width  = input_width  * col_stride
    
    return out_height, out_width

  def _deconv2d(self, x, W, input_width, input_height, stride):
    filter_height = W.get_shape()[0].value
    filter_width  = W.get_shape()[1].value
    out_channel   = W.get_shape()[2].value
    
    out_height, out_width = self._get2d_deconv_output_size(input_height,
                                                           input_width,
                                                           filter_height,
                                                           filter_width,
                                                           stride,
                                                           'VALID')
    batch_size = tf.shape(x)[0]
    output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='VALID')


# ----------------------------------------------------------------------------------------------------------------------
# Utils

  def _for_each(self,input_arrays,op,output_shape):
      output_array = op(*map(lambda input_array: input_array[:1],input_arrays))
      output_array.set_shape(output_shape)
      i0 = tf.constant(1)
      _, output = tf.while_loop(
          lambda i, oa: i < tf.shape(input_arrays[0])[0],
          lambda i, oa: self._concat_op_and_increment(input_arrays,oa,op,i),
          [i0, output_array],
          shape_invariants=[i0.get_shape(),
                            tf.TensorShape(output_shape)])
      return output

  def _concat_op_and_increment(self,input_arrays,output_array,op,inc):
      output_array = tf.concat([output_array,op(*map(lambda input_array: input_array[inc:inc+1], input_arrays))],0)
      return inc+1,output_array
