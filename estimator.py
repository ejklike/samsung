import argparse
from datetime import datetime
import time
import sys
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import dataloader
import func

FLAGS = None

# INFO, DEBUG, ERROR
tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  # Load datasets
  # Get signals and labels
  signals_trn, labels_trn = dataloader.load(which_data=FLAGS.data_type, train=True)
  signals_tst, labels_tst = dataloader.load(which_data=FLAGS.data_type, train=False)

  # signals_trn, signals_val, labels_trn, labels_val = train_test_split(
  #    signals_trn, labels_trn, test_size=0.2, random_state=42, stratify=labels_trn[:, 1])

  # Model dir
  model_dir = './tf_models/%s_model%s' % (FLAGS.data_type, FLAGS.model)
  if FLAGS.restart:
    if tf.gfile.Exists(model_dir):
      tf.gfile.DeleteRecursively(model_dir)
    tf.gfile.MakeDirs(model_dir)

  # Set model params
  model_params = dict(
    learning_rate=FLAGS.learning_rate,
    # regularization type and strength
    wd=FLAGS.wd,
    reg_type=FLAGS.model,
    # convolutional layer
    window_size=5,
    n_conv=1,
    n_filter=30,
    # fully connected layer
    n_fully_connected=1,
    # pooling
    pooling_size=2,
    pooling_stride=1,
  )

  # config for saving things
  config = tf.estimator.RunConfig()
  session_config = tf.ConfigProto(
    log_device_placement=False
  )
  config = config.replace(
    save_summary_steps=100,
    save_checkpoints_steps=1000,
    save_checkpoints_secs=None,
    session_config=session_config,
  )

  # Instantiate Estimator
  estimator = tf.estimator.Estimator(
      model_fn=func.model_fn, 
      params=model_params,
      model_dir=model_dir,
      config=config)

  class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def begin(self):
        self._step = 0

    def before_run(self, run_context):
        self._step += 1
    #    return tf.train.SessionRunArgs(loss_trn)  # Asks for loss value.

    def after_run(self, run_context, run_values):
      if self._step == FLAGS.steps:
        print('>> this is hook!')
        losses = run_context.session.graph.get_collection("losses")
        total_loss = run_context.session.graph.get_collection("total_loss")
        print('>> losses, total_loss = ', run_context.session.run([losses, total_loss]))

  # Input functions
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'signals': signals_trn},
      y=labels_trn,
      batch_size=FLAGS.batch_size,
      num_epochs=None,
      shuffle=True)
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'signals': signals_tst},
      y=labels_tst,
      num_epochs=1,
      shuffle=False)

  # Train and test
  # We iterate train and evaluation to save summaries
  for _ in range(FLAGS.steps // FLAGS.log_freq):
    estimator.train(
        input_fn=train_input_fn,
        steps=FLAGS.log_freq,
        # hooks=[_EarlyStoppingHook()]
    )
    estimator.evaluate(input_fn=test_input_fn)

  class _SaveWeight(tf.train.SessionRunHook):
    def before_run(self, run_context):
      conv_weight = run_context.session.graph.get_tensor_by_name('conv0/weights:0')
      return tf.train.SessionRunArgs(conv_weight)
    
    def after_run(self, run_context, run_values):
      conv_weight = np.squeeze(run_values[0])
      filter_list = np.dsplit(conv_weight, model_params['n_filter'])
      unified_filter = np.vstack(filter_list)
      fname = 'data_type_%s__conv_weight__model%d.csv' % (FLAGS.data_type, FLAGS.model)
      np.savetxt(fname, unified_filter, delimiter=',')

  # Save the test results
  ev = estimator.evaluate(
    input_fn=test_input_fn,
    hooks=[_SaveWeight()])
  func.save_result(ev, FLAGS)


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
      type=int, 
      default=0,
      help='0/1/2/3')
  parser.add_argument(
      '--wd', 
      type=float, 
      default=0.005,
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
      default=3000,
      help='step size')
  parser.add_argument(
      '--log_freq', 
      type=int, 
      default=100,
      help='log frequency')
  parser.add_argument(
      '--restart', 
      type=bool, 
      default=True,
      help='restart the training')




  FLAGS, unparsed = parser.parse_known_args()

  # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_no
  os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.model)

  tf.app.run(main=main) # , argv=[sys.argv[0]] + unparsed