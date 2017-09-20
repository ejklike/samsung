import os
import re
import sys
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Constants describing the training process.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

# Constants for input data manipulation
N_OVERSAMPLE = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_group_weight_decay(name, shape, wd=None, reg_type='l1'):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.contrib.layers.xavier_initializer(dtype=dtype))
  
  if wd is not None:
    window_size, sensor_size, _, n_filter = shape
    squeezed_var = tf.squeeze(var) # remove channel

    if reg_type == 'group1':
      kernel_list = tf.unstack(value=squeezed_var, num=n_filter, axis=2)
      column_list = [tf.unstack(value=k, num=sensor_size, axis=1) for k in kernel_list]
      weights_list = [col for col_list in kernel_list for col in col_list] # flatten
      group_wd = tf.sqrt(tf.constant(window_size, dtype=dtype))

    elif reg_type == 'group2':
      kernel_list = tf.unstack(value=squeezed_var, num=n_filter, axis=2)
      unified_kernel = tf.concat(kernel_list, axis=0)
      weights_list = tf.unstack(value=unified_kernel, num=sensor_size, axis=1)
      group_wd = tf.sqrt(tf.constant(window_size * n_filter, dtype=dtype))

    elif reg_type == 'l1':
      weights_list = [var]
      group_wd = tf.constant(1.0, dtype=dtype)
      
    regularizer = tf.contrib.layers.l1_regularizer(wd * group_wd)
    weight_regularization = tf.contrib.layers.apply_regularization(
      regularizer, weights_list=weights_list)
    tf.add_to_collection('losses', weight_regularization)
  return var

def inputs(data_fname, train=True):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    signals: Signals. 4D tensor of [batch_size, TIME_SIZE, SENSOR_SIZE, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """

  # signal_initializer = tf.placeholder(dtype=signals.dtype, shape=signals.shape)
  # label_initializer = tf.placeholder(dtype=labels.dtype, shape=labels.shape)

  # input_signals = tf.Variable(signal_initializer, trainable=False, collections=[])
  # input_labels = tf.Variable(label_initializer, trainable=False, collections=[])

  raw_data = pickle.load(open(data_fname, 'rb'))
  signals, labels = raw_data['signals'], raw_data['labels']

  # signals = signals.reshape(signals.shape[0], -1)

  # split into trn and tst
  signals_trn, signals_tst, labels_trn, labels_tst = train_test_split(
    signals, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
    stratify=labels[:, 1])
  
  def tf_constant(signals, labels):
    input_signals, input_labels = tf.constant(signals), tf.constant(labels)

    one_channel_shape = list(signals.shape) + [1]
    input_signals = tf.reshape(input_signals, one_channel_shape, 
                              name='reshaped_input_signal')

    if FLAGS.use_fp16:
      input_signals = tf.cast(input_signals, tf.float16)
      input_labels = tf.cast(input_labels, tf.float16)
    
    return input_signals, input_labels

  if train:
    def oversample(X, y, n_oversample):
      fault_idx = np.where(y[:, 1]==1)[0]
      X_fault = X[fault_idx, :]
      y_fault = y[fault_idx, :]
      for _ in range(n_oversample):
        X = np.concatenate((X, X_fault), axis=0)
        y = np.concatenate((y, y_fault), axis=0)
      return X, y

    signals_trn, labels_trn = oversample(signals_trn, labels_trn, N_OVERSAMPLE)

    signals_trn, signals_val, labels_trn, labels_val = train_test_split(
      signals_trn, labels_trn, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
      stratify=labels_trn[:, 1])

    FLAGS.batch_size = signals_trn.shape[0]
    
    input_signals_trn, input_labels_trn = tf_constant(signals_trn, labels_trn)
    input_signals_val, input_labels_val = tf_constant(signals_val, labels_val)
    return input_signals_trn, input_labels_trn, input_signals_val, input_labels_val
    
  else:
    signals_tst, labels_tst = tf_constant(signals_tst, labels_tst)
    return signals_tst, labels_tst


def inference(signals, wd=None, reg_type=None):
  """Build the model.

  Args:
    signals: Signals returned from inputs().

  Returns:
    Logits.
  """

  batch_size, time_size, sensor_size, _ = signals.get_shape().as_list()

  window_size = 7
  n_filter = 20
  n_hidden = 50
  
  pooling_size = 2
  pooling_stride = 2
  padding = 'VALID'
  
  wd = 0.01
  reg_type = 'group2'

  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_group_weight_decay('weights',
                                         shape=[window_size, sensor_size, 1, n_filter],
                                         wd=wd,
                                         reg_type=reg_type)
    conv = tf.nn.conv2d(signals, kernel, [1, 1, 1, 1], padding=padding)
    biases = _variable_on_cpu('biases', [n_filter], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, 
                         ksize=[1, pooling_size, 1, 1], 
                         strides=[1, pooling_stride, 1, 1],
                         padding=padding, name='pool1')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool1, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_group_weight_decay('weights', shape=[dim, n_hidden])
    biases = _variable_on_cpu('biases', [n_hidden], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_group_weight_decay('weights', [n_hidden, 2])
    biases = _variable_on_cpu('biases', [2],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels, test_metric=False):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  # minor_weight = 5
  # weight_vector = 1 + labels[:, 1] * (minor_weight - 1)
  # cross_entropy_mean = tf.losses.softmax_cross_entropy(
  #     onehot_labels=labels,
  #     logits=logits,
  #     weights=weight_vector,
  #     label_smoothing=0,
  #     scope='cross_entropy_weighted',
  #     loss_collection=None
  # )
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  if test_metric is False:
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
  else:    
    def my_metrics(logits, labels):
      true_y, pred_y = tf.argmax(labels, 1), tf.argmax(logits, 1)
      correct_pred = tf.equal(true_y, pred_y)
      incorrect_pred = tf.logical_not(correct_pred)

      # confusion matrix
      #               true 1   true 0
      # predicted 1     TP       FP
      # predicted 0     FN       TN
      tp = tf.reduce_sum(tf.boolean_mask(true_y, correct_pred))
      tn = tf.reduce_sum(tf.boolean_mask(1 - true_y, correct_pred))
      fp = tf.reduce_sum(tf.boolean_mask(1 - true_y, incorrect_pred))
      fn = tf.reduce_sum(tf.boolean_mask(true_y, incorrect_pred))

      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      fscore = 2 / (1/precision + 1/recall)
      return accuracy, precision, recall, fscore
    
    return (
      tf.add_n(tf.get_collection('losses'), name='total_loss'),
      my_metrics(logits, labels)
    )


def _add_loss_summaries(total_loss):
  """Add summaries for losses in model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op