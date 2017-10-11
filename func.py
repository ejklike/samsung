import os

import tensorflow as tf

import reg_wrapper

padding = 'VALID'

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var


def _variable_with_regularization(name, shape, wd=None, reg_type=None):
  """Helper to create an initialized Variable with weight decay."""
  var = _variable_on_cpu(
      name,
      shape,
      tf.contrib.layers.xavier_initializer(dtype=tf.float32))

  if reg_type is not None and reg_type != 0:
    regularizer = {
      1: reg_wrapper.group_regularizer(wd),
      2: reg_wrapper.group_regularizer(wd, unified=True),
      3: tf.contrib.layers.l1_regularizer(wd)
    }[reg_type]
    weight_regularization = tf.contrib.layers.apply_regularization(
      regularizer, weights_list=[var])
    tf.add_to_collection('losses', weight_regularization)
  return var


def _conv_layer(x, shape, wd, reg_type, name=None):
  """conv layer """
  name = 'conv' if name is None else name
  with tf.variable_scope(name) as scope:
    kernel = _variable_with_regularization('weights',
                                           shape=shape,
                                           wd=wd,
                                           reg_type=reg_type)
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding=padding)
    biases = _variable_on_cpu('biases', [shape[-1]], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    activation = tf.nn.relu(pre_activation, name=scope.name)
  return activation


def _pool_layer(x, pooling_size, pooling_stride, name=None):
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
    weights = _variable_with_regularization('weights', shape=[dim, out_dim])
    biases = _variable_on_cpu('biases', [out_dim], tf.constant_initializer(0.1))
    if activation_fn:
      x = activation_fn(tf.matmul(x, weights) + biases, name=scope.name)
    else:
      x = tf.add(tf.matmul(x, weights), biases, name=scope.name)
  return x


def inference(signals, params):
  window_size = params['window_size']
  n_filter = params['n_filter']
  
  n_conv = params['n_conv']
  n_fully_connected = params['n_fully_connected']

  pooling_size = params['pooling_size']
  pooling_stride = params['pooling_stride']

  # input reshaping
  # with tf.device('/cpu:0'):
  _, time_size, sensor_size = signals.get_shape().as_list()
  signals = tf.reshape(signals, 
                      shape=[-1, time_size, sensor_size, 1], 
                      name='reshaped_input_signal')

  # conv0 with weight regularization
  x = _conv_layer(signals, 
                  [window_size, sensor_size, 1, n_filter], 
                  wd=params['wd'], 
                  reg_type=params['reg_type'], 
                  name='conv0') ####
  x = _pool_layer(x, pooling_size, pooling_stride, name='pool0')
  # print('conv0', x.get_shape().as_list())

  # conv1 and more
  for i in range(1, n_conv):
    sensor_size = x.get_shape().as_list()[2]
    x = _conv_layer(x, [window_size, sensor_size, n_filter, n_filter],
                    wd=None, reg_type=None, name='conv%d'%i)
    x = _pool_layer(x, pooling_size, pooling_stride, name='pool%d'%i)
    print('conv%d'%i, x.get_shape().as_list())

  # flatten
  x = _flatten(x, name='flatten')
  # print('flatten size', x.get_shape())
  
  # fully connected
  for i in range(n_fully_connected):
    n_hidden = x.get_shape().as_list()[1] //2
    x = _fully_conn_layer(x, n_hidden, name='fully_conn%d'%i)
  
  # final logit
  logits = _fully_conn_layer(x, 2, activation_fn=None, name='softmax_linear')

  # # check variables
  # for v in tf.trainable_variables():
  #   print(v.name, ':', v.get_shape())

  return logits


def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  signals = features['signals']
  logits = inference(signals, params)

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
  tf.add_to_collection('losses', cross_entropy_mean)
  # print(tf.get_collection('losses'))
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params['learning_rate'])
  train_op = optimizer.minimize(
      loss=total_loss, global_step=tf.train.get_global_step())

  def eval_metrics(logits, labels):
    true_y, pred_y = tf.argmax(labels, 1), tf.argmax(logits, 1)
    acc = tf.metrics.accuracy(true_y, pred_y)
    prec = tf.metrics.precision(true_y, pred_y)
    rec = tf.metrics.recall(true_y, pred_y)
    f1_neg = 2.0 / (1.0/prec[0] + 1.0/rec[0])
    f1_pos = 2.0 / (1.0/prec[1] + 1.0/rec[1])
    f1 = (f1_neg, f1_pos)
    return acc, prec, rec, f1

  acc, prec, rec, f1 = eval_metrics(logits, labels)

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


def save_result(ev, FLAGS):
    fname = './results.csv'
    if not os.path.exists(fname):
      with open(fname, 'w') as fout:
        fout.write('data,model,wd,step,loss,accuracy,precision,recall,averagef1,')
        fout.write('\n')

    with open(fname, 'a') as fout:
        fout.write('{},'.format(FLAGS.data_type))
        fout.write('{},'.format(FLAGS.model))
        fout.write('{},'.format(FLAGS.wd))
        fout.write('{},'.format(tf.train.get_global_step()))
        fout.write('{},'.format(ev['loss']))
        fout.write('{},'.format(ev['accuracy']))
        fout.write('{},'.format(ev['precision']))
        fout.write('{},'.format(ev['recall']))
        fout.write('{},'.format(ev['averagef1']))
        fout.write('\n')
