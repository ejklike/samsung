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
N_OVERSAMPLE = 0
TEST_SIZE = 0.2
RANDOM_STATE = 42

# batch generation
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def _activation_summary(x):
  """Helper to create summaries for activations.
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_group_weight_decay(name, shape, wd=None, reg_type='l1'):
  """Helper to create an initialized Variable with weight decay."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.contrib.layers.xavier_initializer(dtype=dtype))

  if wd is not None:
    regularizer = dict(
      group1=my_tf_func.group_regularizer(wd),
      group2=my_tf_func.group_regularizer(wd, unified=True),
      l1=tf.contrib.layers.l1_regularizer(wd)
    )[reg_type]
    weight_regularization = tf.contrib.layers.apply_regularization(
      regularizer, weights_list=[var])
    tf.add_to_collection('losses', weight_regularization)
  return var


def inputs(data_fname, train=True):
  """Construct input for evaluation using the Reader ops.
  """
  # read data
  # // which data
  # // train or not
  images, labels = my_input.input_batch(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)

  return signals, labels


def inference(signals, wd=None, reg_type=None):
  """Build the model.

  Args:
    signals: Signals returned from inputs().

  Returns:
    Logits.
  """

  batch_size, time_size, sensor_size, _ = signals.get_shape().as_list()

  window_size = 5
  n_filter = 20
  n_hidden = 50
  
  pooling_size = 2
  pooling_stride = 1
  padding = 'VALID'

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