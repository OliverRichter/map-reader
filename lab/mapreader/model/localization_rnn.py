import tensorflow as tf
import sys
import numpy as np


# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


class localizationCell(tf.contrib.rnn.RNNCell):
  def __init__(self,local_map_width,input_size):
    super(localizationCell, self)
    self._input_size = input_size
    self._local_map_width = local_map_width
    self._local_map_size = local_map_width**2
    self._local_map_shape = [1,self._local_map_width,self._local_map_width,1]
    self._local_map_width_extended = local_map_width + 2
    self._local_map_size_extended = self._local_map_width_extended**2

  @property
  def state_size(self):
    return 2*self._local_map_size_extended+9

  @property
  def output_size(self):
    return 63**2+1+9

  def set_map(self,map_63):
    self.map = map_63
    padding_width = self._local_map_width_extended//2
    padded_map = tf.pad(self.map[0,:,:,0],[[padding_width,padding_width],[padding_width,padding_width]])
    all_local_maps = []
    for i in range(63):
      for j in range(63):
        all_local_maps.append(padded_map[i:(i+self._local_map_width_extended),j:(j+self._local_map_width_extended)])
    self.all_local_maps = tf.reshape(all_local_maps,[63**2,self._local_map_size_extended])

  def __call__(self, inputs, state):
    map_state, shift_state = state
    last_action_reward_angle_shift_state = tf.concat([inputs[:,self._local_map_size:],shift_state],1)
    visual_local_map = tf.reshape(inputs[:,:self._local_map_size],[1,self._local_map_width,self._local_map_width,1])
    visual_local_map_filter = tf.reshape(inputs[:,:self._local_map_size],[self._local_map_width,self._local_map_width,1,1])

    W_shift_1, b_shift_1 = self._fc_variable([46,32],"shift_fc_1")
    W_shift_2, b_shift_2 = self._fc_variable([32,9],"shift_fc_2")
    h_shift = tf.nn.relu(tf.matmul(last_action_reward_angle_shift_state,W_shift_1)+b_shift_1)
    shift = tf.matmul(h_shift,W_shift_2)+b_shift_2 + tf.reshape(tf.nn.conv2d(map_state[:,:,:,1:],visual_local_map_filter,[1,1,1,1],"VALID"),[1,9])
    shift_weights = tf.reshape(tf.nn.softmax(shift),[3,3,1,1])
    shifted_map_feedback = tf.nn.conv2d(map_state[:,:,:,:1],shift_weights,[1,1,1,1],"VALID")
    shifted_previously_estimated_local_map = tf.nn.conv2d(map_state[:,:,:,1:],shift_weights,[1,1,1,1],"VALID")

    lam = tf.get_variable('lambda',initializer=0.1)
    estimated_local_map = tf.clip_by_value(shifted_previously_estimated_local_map + visual_local_map,-0.5,0.5)
    estimated_local_map_with_map_feedback = tf.clip_by_value(estimated_local_map + lam*shifted_map_feedback,-0.5,0.5)

    location_filter = tf.reshape(estimated_local_map_with_map_feedback,[self._local_map_width,self._local_map_width,1,1])

    location_believe = tf.nn.conv2d(self.map,location_filter,[1,1,1,1],"SAME")
    location_believe_ = tf.reshape(location_believe,[1,63**2])
    location_believe_distribution = tf.clip_by_value(tf.nn.softmax(location_believe_),1e-20,1.0)

    #normalized entropy
    location_uncertainty = tf.stop_gradient(-tf.reduce_sum(tf.multiply(location_believe_distribution,tf.log(location_believe_distribution))))/tf.log(63.0**2)

    map_feedback = tf.matmul(location_believe_distribution,self.all_local_maps)
    shift_weights_ = tf.reshape(shift_weights,[1,9])
    localization_output = tf.concat([location_believe_,tf.reshape(location_uncertainty,[1,1]),shift_weights_],1)
    map_feedback_ = tf.reshape(map_feedback,[1, self._local_map_width+2,self._local_map_width+2,1])
    localization_state = (tf.concat([map_feedback_,tf.pad(estimated_local_map,[[0,0],[1,1],[1,1],[0,0]])],3),shift_weights_)
    return localization_output, localization_state

  def _fc_variable(self, weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)

    input_channels = weight_shape[0]
    output_channels = weight_shape[1]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias = tf.get_variable(name_b, bias_shape, initializer=fc_initializer(input_channels))
    return weight, bias