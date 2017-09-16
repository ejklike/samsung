import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, static_rnn

from tensorflow.python.ops.rnn_cell import LSTMCell
import numpy as np


###
# Recurrent neural network
###
def rnn(x, param_dict, FLAGS):
    """
    input: placeholder with shape [batch_size, time_size, sensor_size]
    output: lstm last output --> fully_connected --> linear layer output
    """
    time_size = FLAGS.time_size
    sensor_size = FLAGS.sensor_size
    
    n_hidden = param_dict.n_hidden
    n_rnn_unit = param_dict.n_rnn_unit

    # unstack --> time_size * [batch_size, sensor_size]
    with tf.device('/cpu:0'):
        x = tf.unstack(x, time_size, 1)
    
    # Define a lstm cell with tensorflow
    cell = BasicLSTMCell(n_rnn_unit, forget_bias=1.0)
    # cell = BasicRNNCell(n_rnn_unit)

    # Get lstm cell output
    with tf.variable_scope('lstm'):
        outputs, states = static_rnn(cell, x, dtype=tf.float32)
    
    rnn_output = outputs[-1]
    
    if n_hidden == 0:
        return layers.linear(rnn_output, 2)
    else:
        fully_conn = layers.fully_connected(rnn_output, n_hidden)
        return layers.linear(fully_conn, 2)

###
# Recurrent neural network
###
def lstm(x, param_dict, FLAGS):
    """
    input: placeholder with shape [batch_size, time_size, sensor_size]
    output: lstm last output --> fully_connected --> linear layer output
    """
    time_size = FLAGS.time_size
    sensor_size = FLAGS.sensor_size
    
    n_hidden = param_dict.n_hidden
    n_rnn_unit = param_dict.n_rnn_unit

    # unstack --> time_size * [batch_size, sensor_size]
    with tf.device('/cpu:0'):
        x = tf.unstack(x, time_size, 1)
    
    # Define a lstm cell with tensorflow
    # cell = BasicLSTMCell(n_rnn_unit, forget_bias=1.0)
    cell = BasicRNNCell(n_rnn_unit)

    # Get lstm cell output
    with tf.variable_scope('lstm'):
        outputs, states = static_rnn(cell, x, dtype=tf.float32)
    
    rnn_output = outputs[-1]
    
    if n_hidden == 0:
        return layers.linear(rnn_output, 2)
    else:
        fully_conn = layers.fully_connected(rnn_output, n_hidden)
        return layers.linear(fully_conn, 2)

###
# convolutional neural network
###
def cnn(x, param_dict, FLAGS):
    """
    input: placeholder with shape [batch_size, time_size, sensor_size]
    output: cnn output --> fully_connected --> linear layer output
    """
    time_size = FLAGS.time_size
    sensor_size = FLAGS.sensor_size

    window_size = param_dict.window_size
    n_filter = param_dict.n_filter
    pooling_size = param_dict.pooling_size
    n_hidden = param_dict.n_hidden

    # [batch_size, time_size, sensor_size, channels]
    with tf.device('/cpu:0'):
        _x_reshaped = tf.reshape(x, [-1, time_size, sensor_size, 1], name='reshaped_x')
    # convolutional layer
    conv = layers.conv2d(
        inputs=_x_reshaped,
        kernel_size=[window_size, sensor_size],
        num_outputs=n_filter, 
        stride=1,
        padding='VALID'
        )
    # pooling layer
    pool = layers.max_pool2d(
        inputs=conv,
        kernel_size=[pooling_size, 1],
        stride=1)

    # pool_reshaped = tf.reshape(pool, 
    #     [-1, time_size - window_size + 1 - pooling_size + 1, n_filter, 1])
    # # convolutional layer
    # conv = layers.conv2d(
    #     inputs=pool_reshaped,
    #     kernel_size=[window_size, n_filter],
    #     num_outputs=n_filter, 
    #     stride=1,
    #     padding='VALID'
    # )
    # # pooling layer
    # pool = layers.max_pool2d(
    #     inputs=conv,
    #     kernel_size=[pooling_size, 1],
    #     stride=1) #[pooling_size//2, 1]

    # flatten layer
    flat = layers.flatten(pool, scope='flatten')
    if n_hidden == 0:
        # print(_x_reshaped.shape, conv.shape, pool.shape, flat.shape)
        return layers.linear(flat, 2, scope='final_node')
    else:
        fully_conn = layers.fully_connected(flat, n_hidden, scope='fully_conn')
        # print(_x_reshaped.shape, conv.shape, pool.shape, flat.shape, fully_conn.shape)
        return layers.linear(fully_conn, 2, scope='final_node')

















# ###
# # convolutional neural network
# ###
# def nnnnn(x, param_dict, FLAGS):
#     """
#     input: placeholder with shape [batch_size, time_size, sensor_size]
#     output: cnn output --> fully_connected --> linear layer output
#     """
#     time_size = FLAGS.time_size
#     sensor_size = FLAGS.sensor_size

#     window_size = param_dict.window_size
#     n_filter = param_dict.n_filter
#     pooling_size = param_dict.pooling_size
#     n_hidden = param_dict.n_hidden

#     # [batch_size, time_size, sensor_size, channels]
#     with tf.device('/cpu:0'):
#         _x_reshaped = tf.reshape(x, [-1, time_size, sensor_size, 1], name='reshaped_x')
#     # convolutional layer
#     conv = layers.conv2d(
#         inputs=_x_reshaped,
#         kernel_size=[window_size, sensor_size],
#         num_outputs=n_filter, 
#         stride=1,
#         padding='VALID'
#         )
#     # pooling layer
#     pool = layers.max_pool2d(
#         inputs=conv,
#         kernel_size=[pooling_size, 1],
#         stride=1)

#     # pool_reshaped = tf.reshape(pool, 
#     #     [-1, time_size - window_size + 1 - pooling_size + 1, n_filter, 1])
#     # # convolutional layer
#     # conv = layers.conv2d(
#     #     inputs=pool_reshaped,
#     #     kernel_size=[window_size, n_filter],
#     #     num_outputs=n_filter, 
#     #     stride=1,
#     #     padding='VALID'
#     # )
#     # # pooling layer
#     # pool = layers.max_pool2d(
#     #     inputs=conv,
#     #     kernel_size=[pooling_size, 1],
#     #     stride=1) #[pooling_size//2, 1]

#     # flatten layer
#     flat = layers.flatten(pool, scope='flatten')
#     if n_hidden == 0:
#         # print(_x_reshaped.shape, conv.shape, pool.shape, flat.shape)
#         return layers.linear(flat, 2, scope='final_node')
#     else:
#         fully_conn = layers.fully_connected(flat, n_hidden, scope='fully_conn')
#         # print(_x_reshaped.shape, conv.shape, pool.shape, flat.shape, fully_conn.shape)
#         return layers.linear(fully_conn, 2, scope='final_node')



# ###
# # convolutional neural network --> sensor by sensor
# ###
# def individual_cnn(x, param_dict, FLAGS):
#     """
#     input: placeholder with shape [batch_size, time_size, sensor_size]
#     output: cnn output --> fully_connected --> linear layer output
#     """
#     time_size = FLAGS.time_size
#     sensor_size = FLAGS.sensor_size

#     window_size = param_dict.window_size
#     n_filter = param_dict.n_filter
#     pooling_size = param_dict.pooling_size
#     n_hidden = param_dict.n_hidden

#     # [batch_size, time_size, sensor_size, channels]
#     with tf.device('/cpu:0'):
#         _x_reshaped = tf.reshape(x, [-1, time_size, sensor_size, 1], name='reshaped_x')
#         _x_split = tf.unstack(x, sensor_size, 2)
#     # convolutional layer for each sensor
#     conv_list = [layers.conv2d(
#         inputs=_x_split[i],
#         kernel_size=[window_size, 1],
#         num_outputs=1, 
#         stride=1,
#         padding='VALID') for i in range(sensor_size)]
#     # pooling layer
#     pool_list = [layers.max_pool2d(
#         inputs=conv,
#         kernel_size=[pooling_size, 1],
#         stride=pooling_size//2) for conv in conv_list]
#     # flatten layer
#     flat = layers.flatten(pool_list, scope='flatten')


    
#     if n_hidden == 0:
#         # print(_x_reshaped.shape, conv.shape, pool.shape, flat.shape)
#         return layers.linear(flat, 2, scope='final_node')
#     else:
#         fully_conn = layers.fully_connected(flat, n_hidden, scope='fully_conn')
#         # print(_x_reshaped.shape, conv.shape, pool.shape, flat.shape, fully_conn.shape)
#         return layers.linear(fully_conn, 2, scope='final_node')


# ###
# # reulgarization + convolutional neural network
# ###
# import numbers

# from tensorflow.python.framework import ops
# from tensorflow.python.ops import standard_ops
# from tensorflow.python.platform import tf_logging as logging

# def my_regularizer(scale, scope=None):
#   """Returns a function that can be used to apply L1 regularization to weights.
#   L1 regularization encourages sparsity.
#   Args:
#     scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
#     scope: An optional scope name.
#   Returns:
#     A function with signature `l1(weights)` that apply L1 regularization.
#   Raises:
#     ValueError: If scale is negative or if scale is not a float.
#   """
#   if isinstance(scale, numbers.Integral):
#     raise ValueError('scale cannot be an integer: %s' % scale)
#   if isinstance(scale, numbers.Real):
#     if scale < 0.:
#       raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
#                        scale)
#     if scale == 0.:
#       logging.info('Scale of 0 disables regularizer.')
#       return lambda _: None

#   def my(weights, name=None):
#     """spectral_norm"""
#     with ops.name_scope(scope, 'l1_regularizer', [weights]) as name:
#       my_scale = ops.convert_to_tensor(scale,
#                                        dtype=weights.dtype.base_dtype,
#                                        name='scale')

#       reduced_weights = standard_ops.reduce_sum(
#         weights**2, axis=0
#       )
#       my_ftn = standard_ops.reduce_sum(standard_ops.sqrt(reduced_weights))
#       return standard_ops.multiply(
#           my_scale,
#           my_ftn,
#           name=name)

#       # return standard_ops.multiply(
#       #     my_scale,
#       #     standard_ops.reduce_sum(standard_ops.abs(weights)),
#       #     name=name)

#   return my


# def reg_cnn(x, param_dict, FLAGS):
#     """
#     input: placeholder with shape [batch_size, time_size, sensor_size]
#     output: cnn output --> fully_connected --> linear layer output
#     """
#     time_size = FLAGS.time_size
#     sensor_size = FLAGS.sensor_size

#     window_size = param_dict.window_size
#     n_filter = param_dict.n_filter
#     pooling_size = param_dict.pooling_size
#     n_hidden = param_dict.n_hidden

#     # [batch_size, time_size, sensor_size, channels]
#     with tf.device('/cpu:0'):
#         _x_reshaped = tf.reshape(x, [-1, time_size, sensor_size, 1], name='reshaped_x')
#     # convolutional layer
#     conv = layers.conv2d(
#         inputs=_x_reshaped,
#         kernel_size=[window_size, sensor_size],
#         num_outputs=n_filter, 
#         stride=1,
#         padding='VALID',
#         weights_regularizer=my_regularizer
#         )
#     # pooling layer
#     pool = layers.max_pool2d(
#         inputs=conv,
#         kernel_size=[pooling_size, 1],
#         stride=pooling_size//2)
#     # flatten layer
#     flat = layers.flatten(pool, scope='flatten')
    
#     if n_hidden == 0:
#         # print(_x_reshaped.shape, conv.shape, pool.shape, flat.shape)
#         return layers.linear(flat, 2, scope='final_node')
#     else:
#         fully_conn = layers.fully_connected(flat, n_hidden, scope='fully_conn')
#         # print(_x_reshaped.shape, conv.shape, pool.shape, flat.shape, fully_conn.shape)
#         return layers.linear(fully_conn, 2, scope='final_node')



# ###
# # LSTMAutoencoder
# ###

# """
# Future : Modularization
# """

# class LSTMAutoencoder(object):
#   """Basic version of LSTM-autoencoder.
#   (cf. http://arxiv.org/abs/1502.04681)
#   Usage:
#     ae = LSTMAutoencoder(hidden_num, inputs)
#     sess.run(ae.train)
#   """

#   def __init__(self, hidden_num, inputs, 
#     cell=None, optimizer=None, reverse=True, 
#     decode_without_input=False):
#     """
#     Args:
#       hidden_num : number of hidden elements of each LSTM unit.
#       inputs : a list of input tensors with size 
#               (batch_num x elem_num)
#       cell : an rnn cell object (the default option 
#             is `tf.python.ops.rnn_cell.LSTMCell`)
#       optimizer : optimizer for rnn (the default option is
#               `tf.train.AdamOptimizer`)
#       reverse : Option to decode in reverse order.
#       decode_without_input : Option to decode without input.
#     """

#     self.batch_num = inputs[0].get_shape().as_list()[0]
#     self.elem_num = inputs[0].get_shape().as_list()[1]

#     if cell is None:
#       self._enc_cell = LSTMCell(hidden_num)
#       self._dec_cell = LSTMCell(hidden_num)
#     else :
#       self._enc_cell = cell
#       self._dec_cell = cell

#     with tf.variable_scope('encoder'):
#       self.z_codes, self.enc_state = tf.nn.rnn(
#         self._enc_cell, inputs, dtype=tf.float32)

#     with tf.variable_scope('decoder') as vs:
#       dec_weight_ = tf.Variable(
#         tf.truncated_normal([hidden_num, self.elem_num], dtype=tf.float32),
#         name="dec_weight")
#       dec_bias_ = tf.Variable(
#         tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32),
#         name="dec_bias")

#       if decode_without_input:
#         dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
#                       for _ in range(len(inputs))]
#         dec_outputs, dec_state = tf.nn.rnn(
#           self._dec_cell, dec_inputs, 
#           initial_state=self.enc_state, dtype=tf.float32)
#         """the shape of each tensor
#           dec_output_ : (step_num x hidden_num)
#           dec_weight_ : (hidden_num x elem_num)
#           dec_bias_ : (elem_num)
#           output_ : (step_num x elem_num)
#           input_ : (step_num x elem_num)
#         """
#         if reverse:
#           dec_outputs = dec_outputs[::-1]
#         dec_output_ = tf.transpose(tf.pack(dec_outputs), [1,0,2])
#         dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [self.batch_num,1,1])
#         self.output_ = tf.batch_matmul(dec_output_, dec_weight_) + dec_bias_

#       else : 
#         dec_state = self.enc_state
#         dec_input_ = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
#         dec_outputs = []
#         for step in range(len(inputs)):
#           if step>0: vs.reuse_variables()
#           dec_input_, dec_state = self._dec_cell(dec_input_, dec_state)
#           dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
#           dec_outputs.append(dec_input_)
#         if reverse:
#           dec_outputs = dec_outputs[::-1]
#         self.output_ = tf.transpose(tf.pack(dec_outputs), [1,0,2])

#     self.input_ = tf.transpose(tf.pack(inputs), [1,0,2])
#     self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))

#     if optimizer is None :
#       self.train = tf.train.AdamOptimizer().minimize(self.loss)
#     else :
#       self.train = optimizer.minimize(self.loss)