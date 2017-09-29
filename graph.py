import os
import re
import sys

import numpy as np
import tensorflow as tf

import dataloader
import reg_wrapper

FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
DTYPE = tf.float32


def _activation_summary(x):
  tf.summary.histogram(x.op.name + '/activations', x)
  tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    # var = tf.get_variable(name, shape, initializer=initializer, dtype=FLAGS.dtype)
    var = tf.get_variable(name, shape, initializer=initializer, dtype=DTYPE)
  return var


def _variable_with_group_weight_decay(name, shape, wd=None, reg_type='l1', 
                                      add_loss=False):
  """Helper to create an initialized Variable with weight decay."""
  var = _variable_on_cpu(
      name,
      shape,
      # tf.contrib.layers.xavier_initializer(dtype=FLAGS.dtype))
      tf.contrib.layers.xavier_initializer(dtype=DTYPE))

  if wd is not None and add_loss:
    regularizer = dict(
      group1=reg_wrapper.group_regularizer(wd),
      group2=reg_wrapper.group_regularizer(wd, unified=True),
      l1=tf.contrib.layers.l1_regularizer(wd)
    )[reg_type]
    weight_regularization = tf.contrib.layers.apply_regularization(
      regularizer, weights_list=[var])
    tf.add_to_collection('losses', weight_regularization)
  return var
  

def inputs(signals, labels, test=False):
  # Input data, pin to CPU because rest of pipeline is CPU-only
  with tf.device('/cpu:0'):
    signals = tf.constant(signals, dtype=FLAGS.dtype)
    labels = tf.constant(labels, dtype=FLAGS.dtype)
    # print(input_signals, input_labels)

    one_channel_shape = signals.get_shape().as_list() + [1]
    signals = tf.reshape(signals, one_channel_shape, 
                         name='reshaped_input_signal')

  # the maximum number of elements in the queue
  capacity = 20 * FLAGS.batch_size

  # params to be determined by trn or tst
  num_epochs = 1 if test else FLAGS.num_epochs
  allow_smaller_final_batch = True if test else False

  signal, label = tf.train.slice_input_producer(
      [signals, labels], num_epochs=num_epochs)

  signals, labels = tf.train.batch(
      [signal, label], 
      batch_size=FLAGS.batch_size, 
      num_threads=FLAGS.num_threads,
      capacity=capacity,
      allow_smaller_final_batch=allow_smaller_final_batch)

  return signals, labels


def inference(signals, add_loss=False):
  _, _, sensor_size, _ = signals.get_shape().as_list()

  window_size = 5
  n_filter = 50
  n_hidden = 100
  
  pooling_size = 2
  pooling_stride = 1
  padding = 'VALID'

  def _conv_layer(x, shape, wd, reg_type, name=None):
    """conv layer """
    name = 'conv' if name is None else name
    with tf.variable_scope(name) as scope:
      kernel = _variable_with_group_weight_decay('weights',
                      shape=shape,
                      wd=wd,
                      reg_type=reg_type,
                      add_loss=add_loss)
      conv = tf.nn.conv2d(signals, kernel, [1, 1, 1, 1], padding=padding)
      biases = _variable_on_cpu('biases', [shape[-1]], tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      activation = tf.nn.relu(pre_activation, name=scope.name)
      _activation_summary(activation)
    return activation

  def _pool_layer(x, name=None):
    name = 'pool' if name is None else name
    with tf.name_scope(name) as name_scope:
      x = tf.nn.max_pool(x, ksize=[1, pooling_size, 1, 1], 
                         strides=[1, pooling_stride, 1, 1],
                         padding=padding, name='pool1')
    return x

  def _flatten(x, name=None):
    name = 'flatten' if name is None else name
    # x = tf.reshape(x, [FLAGS.batch_size, -1])
    _, n_time, n_sensor, n_filter = x.get_shape().as_list()
    x = tf.reshape(x, [-1, n_time * n_sensor * n_filter], name='flatten')
    return x

  def _fully_conn_layer(x, out_dim, activation_fn=tf.nn.relu, name=None):
    name = 'fully_connected' if name is None else name
    dim = x.get_shape()[-1]
    with tf.variable_scope(name) as scope:
      weights = _variable_with_group_weight_decay('weights', shape=[dim, out_dim])
      biases = _variable_on_cpu('biases', [out_dim], tf.constant_initializer(0.1))
      if activation_fn:
        x = activation_fn(tf.matmul(x, weights) + biases, name=scope.name)
      else:
        x = tf.add(tf.matmul(x, weights), biases, name=scope.name)
      _activation_summary(x)
    return x

  x = _conv_layer(signals, [window_size, sensor_size, 1, n_filter], 
                  wd=FLAGS.wd, reg_type=FLAGS.reg_type, name='conv1') ####
  print('conv1', x.get_shape())
  x = _pool_layer(x, name='pool1')
  print('pool1', x.get_shape())
  x = _flatten(x, name='flatten')
  print('flatten', x.get_shape())
  x = _fully_conn_layer(x, n_hidden, name='fully_conn1')
  print('fully_conn1', x.get_shape())
  return _fully_conn_layer(x, 2, activation_fn=None, name='softmax_linear')


def loss(logits, labels, add_loss=False):
  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
  if add_loss:
    tf.add_to_collection('losses', cross_entropy_mean)
    print(tf.get_collection('losses'))
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  reg_loss = total_loss - cross_entropy_mean
  return total_loss, cross_entropy_mean, reg_loss

def train(total_loss, global_step):
  opt = tf.train.AdamOptimizer(FLAGS.lr)
  train_op = opt.minimize(total_loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  true_y, pred_y = tf.argmax(labels, 1), tf.argmax(logits, 1)
  acc = tf.metrics.accuracy(true_y, pred_y)
  prec = tf.metrics.precision(true_y, pred_y)
  rec = tf.metrics.recall(true_y, pred_y)
  f1_neg = 2.0 / (1.0/prec[0] + 1.0/rec[0])
  f1_pos = 2.0 / (1.0/prec[1] + 1.0/rec[1])
  f1 = (f1_neg, f1_pos)
  return acc, prec, rec, f1