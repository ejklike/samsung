import argparse
from datetime import datetime
import time
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import dataloader
import graph
import reg_wrapper

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)


def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var


def _variable_with_group_weight_decay(name, shape, wd=None, reg_type='0'):
  """Helper to create an initialized Variable with weight decay."""
  var = _variable_on_cpu(
      name,
      shape,
      tf.contrib.layers.xavier_initializer(dtype=tf.float32))

  if reg_type != '0':
    regularizer = {
      '1': reg_wrapper.group_regularizer(wd),
      '2': reg_wrapper.group_regularizer(wd, unified=True),
      '3': tf.contrib.layers.l1_regularizer(wd)
    }[reg_type]
    weight_regularization = tf.contrib.layers.apply_regularization(
      regularizer, weights_list=[var])
    tf.add_to_collection('losses', weight_regularization)
  return var

def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  signals = features['signals']
  _, time_size, sensor_size = signals.get_shape().as_list()
  signals = tf.reshape(signals, 
                       shape=[-1, time_size, sensor_size, 1], 
                       name='reshaped_input_signal')
  

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
                      reg_type=reg_type)
      conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding=padding)
      biases = _variable_on_cpu('biases', [shape[-1]], tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      activation = tf.nn.relu(pre_activation, name=scope.name)
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
    return x

  print('input', signals.get_shape())
  x = _conv_layer(signals, [window_size, sensor_size, 1, n_filter], 
                  wd=FLAGS.wd, reg_type=FLAGS.model, name='conv1') ####
  print('conv1', x.get_shape())
  x = _pool_layer(x, name='pool1')
  print('pool1', x.get_shape())
  sensor_size = x.get_shape().as_list()[2]
  x = _conv_layer(x, [window_size, sensor_size, n_filter, n_filter],
                  wd=None, reg_type='0', name='conv2')
  print('conv2', x.get_shape())
  x = _pool_layer(x, name='pool2')
  print('pool2', x.get_shape())
  x = _flatten(x, name='flatten')
  print('flatten', x.get_shape())
  x = _fully_conn_layer(x, n_hidden, name='fully_conn1')
  print('fully_conn1', x.get_shape())
  x = _fully_conn_layer(x, n_hidden//2, name='fully_conn2')
  print('fully_conn2', x.get_shape())
  logits = _fully_conn_layer(x, 2, activation_fn=None, name='softmax_linear')
  print('softmax_lienar', logits.get_shape())

  for v in tf.trainable_variables():
    print(v.name, ':', v.get_shape())

  # Provide an estimator spec for `ModeKeys.PREDICT`.
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"ages": predictions})

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
  tf.add_to_collection('losses', cross_entropy_mean)
  print(tf.get_collection('losses'))
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params['learning_rate'])
  train_op = optimizer.minimize(
      loss=total_loss, global_step=tf.train.get_global_step())

  def evaluation(logits, labels):
    true_y, pred_y = tf.argmax(labels, 1), tf.argmax(logits, 1)
    acc = tf.metrics.accuracy(true_y, pred_y)
    prec = tf.metrics.precision(true_y, pred_y)
    rec = tf.metrics.recall(true_y, pred_y)
    f1_neg = 2.0 / (1.0/prec[0] + 1.0/rec[0])
    f1_pos = 2.0 / (1.0/prec[1] + 1.0/rec[1])
    f1 = (f1_neg, f1_pos)
    return acc, prec, rec, f1

  acc, prec, rec, f1 = evaluation(logits, labels)

  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = {
      'accuracy': acc,
      'precision': prec,
      'recall': rec,
      'averagef1': f1
  }

  # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


def save_result(ev):
    fname = './results.csv'
    if not os.path.exists(fname):
      with open(fname, 'w') as fout:
        fout.write('data,model,wd,loss,accuracy,precision,recall,averagef1,')
        fout.write('\n')

    with open(fname, 'a') as fout:
        fout.write('{},'.format(FLAGS.data_type))
        fout.write('{},'.format(FLAGS.model))
        fout.write('{},'.format(FLAGS.wd))
        fout.write('{},'.format(ev['loss']))
        fout.write('{},'.format(ev['accuracy']))
        fout.write('{},'.format(ev['precision']))
        fout.write('{},'.format(ev['recall']))
        fout.write('{},'.format(ev['averagef1']))
        fout.write('\n')


def main(unused_argv):
  # Load datasets
  # Get signals and labels
  signals_trn, labels_trn = dataloader.load(which_data=FLAGS.data_type, train=True)
  signals_tst, labels_tst = dataloader.load(which_data=FLAGS.data_type, train=False)

  #signals_trn, signals_val, labels_trn, labels_val = train_test_split(
  #    signals_trn, labels_trn, test_size=0.2, random_state=42, stratify=labels_trn[:, 1])

#   training_set = np.genfromtxt(abalone_train, delimiter=',')
#   test_set = np.genfromtxt(abalone_test, delimiter=',')
#   prediction_set = np.genfromtxt(abalone_predict, delimiter=',')

  # Set model params
  model_params = dict(
    learning_rate=FLAGS.learning_rate
  )

  # Model dir
  model_dir = './tf_models/%s_model%s' % (FLAGS.data_type, FLAGS.model)
  if FLAGS.restart:
    if tf.gfile.Exists(model_dir):
      tf.gfile.DeleteRecursively(model_dir)
    tf.gfile.MakeDirs(model_dir)

  # Instantiate Estimator
  estimator = tf.estimator.Estimator(
      model_fn=model_fn, 
      params=model_params,
      model_dir=model_dir)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'signals': signals_trn},
      y=labels_trn,
      batch_size=FLAGS.batch_size,
      num_epochs=None,
      shuffle=True)

  # Train
  estimator.train(input_fn=train_input_fn, steps=FLAGS.steps)

  # Score accuracy
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'signals': signals_tst},
      y=labels_tst,
      num_epochs=1,
      shuffle=False)

  ev = estimator.evaluate(input_fn=test_input_fn)

  save_result(ev)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  
  # choosing data
  parser.add_argument(
      'data_type', 
      type=str, 
      default='samsung',
      help='choosing data: opp_s#/samsung')
  
  # gpu allocation
  parser.add_argument(
      '--gpu_no', 
      type=str, 
      default='3',
      help='gpu device number')

  # loss parameters
  parser.add_argument(
      '--model', 
      type=str, 
      default='0',
      help='0/1/2/3')
  parser.add_argument(
      '--wd', 
      type=float, 
      default=0.05,
      help='weight decaying factor')

  # learning parameters
  parser.add_argument(
      '--learning_rate', 
      type=float, 
      default=0.0005,
      help='initial learning rate')
  parser.add_argument(
      '--batch_size', 
      type=int, 
      default=500,
      help='batch size')
  parser.add_argument(
      '--steps', 
      type=int, 
      default=10000,
      help='step size')
  parser.add_argument(
      '--restart', 
      type=bool, 
      default=False,
      help='restart the training')


  # parser.add_argument(
  #   '--log_freq', 
  #   type=int, 
  #   default=100,
  #   help='log frequency')

  FLAGS, unparsed = parser.parse_known_args()

  # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_no
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.model

  tf.app.run(main=main) # , argv=[sys.argv[0]] + unparsed