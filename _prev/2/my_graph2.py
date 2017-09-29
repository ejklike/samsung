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

def _activation_summary(x):
  """Helper to create summaries for activations."""
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def group_regularizer(scale, scope=None, unified=False):
  """Returns a function that can be used to apply group regularization to weights."""
  import numbers

  from tensorflow.python.framework import ops
  from tensorflow.python.ops import standard_ops

  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def group_l1(weights):
    """Applies group l1 regularization to weights."""
    window_size, sensor_size, _, n_filter = weights.get_shape().as_list()
    
    with tf.device('/cpu:0'):
      weights = tf.squeeze(weights) # remove channel
      kernel_list = tf.unstack(value=weights, num=n_filter, axis=2)
      axis = 0 if unified else 1 ##
      unified_kernel = tf.concat(kernel_list, axis=axis)
    
    with ops.name_scope(scope, 'group_regularizer', [weights]) as name:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      weight_regularization = tf.reduce_sum(
        tf.sqrt(tf.reduce_sum(unified_kernel**2, axis=0)))
      return standard_ops.multiply(my_scale, weight_regularization, name=name)
  return group_l1

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory."""
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
    window_size, sensor_size, _, n_filter = shape

    if reg_type == 'group1':
      regularizer = group_regularizer(wd)
    elif reg_type == 'group2':
      regularizer = group_regularizer(wd, unified=True)
    elif reg_type == 'l1':
      regularizer = tf.contrib.layers.l1_regularizer(wd)

    weight_regularization = tf.contrib.layers.apply_regularization(
      regularizer, weights_list=[var])
    tf.add_to_collection('losses', weight_regularization)
  return var

def inputs(data_fname, train=True, val=True):
  """Construct input for evaluation using the Reader ops."""

  raw_data = pickle.load(open(data_fname, 'rb'))
  signals, labels = raw_data['signals'], raw_data['labels']

#   raw_data = pickle.load(open('./PM2.p', 'rb'))
#   signals2, labels2 = raw_data['signals'], raw_data['labels']

#   raw_data = pickle.load(open('./PM6.p', 'rb'))
#   signals3, labels3 = raw_data['signals'], raw_data['labels']

#   signals = np.concatenate([signals, signals2, signals3], axis=0)
#   labels = np.concatenate([labels, labels2, labels3], axis=0)

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

    if val:
      signals_trn, signals_val, labels_trn, labels_val = train_test_split(
        signals_trn, labels_trn, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        stratify=labels_trn[:, 1])

      FLAGS.batch_size = signals_trn.shape[0]
      
      input_signals_trn, input_labels_trn = tf_constant(signals_trn, labels_trn)
      input_signals_val, input_labels_val = tf_constant(signals_val, labels_val)
      return input_signals_trn, input_labels_trn, input_signals_val, input_labels_val
    
    else:
      FLAGS.batch_size = signals_trn.shape[0]
      input_signals_trn, input_labels_trn = tf_constant(signals_trn, labels_trn)
      return input_signals_trn, input_labels_trn
    
  else:
    signals_tst, labels_tst = tf_constant(signals_tst, labels_tst)
    return signals_tst, labels_tst

def _generate_signal_and_label_batch(signal, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    signals, label_batch = tf.train.shuffle_batch(
        [signal, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    signals, label_batch = tf.train.batch(
        [signal, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('signals', signals)

  return signals, tf.reshape(label_batch, [batch_size])

def inputs_har(data_fname, train=True, val=True):
  """Construct input for evaluation using the Reader ops."""

  raw_data = pickle.load(open(data_fname, 'rb'))
  signals = raw_data['signals']
  labels_loc, labels_ges = raw_data['labels_locomotion'], raw_data['labels_gesture']

  labels = labels_loc[:, 0:2]

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
    else:
      input_signals = tf.cast(input_signals, tf.float32)
      input_labels = tf.cast(input_labels, tf.float32)

    return input_signals, input_labels

  if train:
    if val:
      signals_trn, signals_val, labels_trn, labels_val = train_test_split(
        signals_trn, labels_trn, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        stratify=labels_trn[:, 1])

      FLAGS.batch_size = signals_trn.shape[0]
      
      input_signals_trn, input_labels_trn = tf_constant(signals_trn, labels_trn)
      input_signals_val, input_labels_val = tf_constant(signals_val, labels_val)
      return input_signals_trn, input_labels_trn, input_signals_val, input_labels_val
    
    else:
      FLAGS.batch_size = signals_trn.shape[0]
      # return input_signals_trn, input_labels_trn
      
      return _generate_signal_and_label_batch(input_signals_trn, input_labels_trn, 
                                    min_queue_examples, batch_size, shuffle)
    
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