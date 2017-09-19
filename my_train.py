import argparse
from datetime import datetime
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from tensorflow.python.client import timeline

import my_graph

FLAGS = tf.app.flags.FLAGS

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
    
    return tp, tn, fp, fn, accuracy, precision, recall, fscore

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      signals_trn, labels_trn, signals_val, labels_val = my_graph.inputs(
        FLAGS.data_fname, train=True)
      signals_tst, labels_tst = my_graph.inputs(
        FLAGS.data_fname, train=False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    with tf.variable_scope('inference') as scope:
      logits_trn = my_graph.inference(signals_trn)
      scope.reuse_variables()
      logits_val = my_graph.inference(signals_val)
      scope.reuse_variables()
      logits_tst = my_graph.inference(signals_tst)

    # Calculate loss.
    loss_trn = my_graph.loss(logits_trn, labels_trn)
    loss_val = my_graph.loss(logits_val, labels_val)
    loss_tst = my_graph.loss(logits_tst, labels_tst)

    # test scores
    tp, tn, fp, fn, acc, prec, rec, f1 = my_metrics(logits_tst, labels_tst)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = my_graph.train(loss_trn, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss_trn)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    class _EarlyStoppingHook(tf.train.SessionRunHook):
      """Early Stopping Hook"""

      def begin(self):
        self._step = -1
        self._tolerance_count = 0
        self._tolerance_max = 100
        self._prev_loss_val_value = 1e10

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss_val)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          loss_val_value = run_values.results
          if loss_val_value > self._prev_loss_val_value:
            self._tolerance_count += 1
          else:
            self._tolerance_count = 0
          self._prev_loss_val_value = loss_val_value
          
          # stop after maximum tolerance
          if self._tolerance_count >= self._tolerance_max:
            run_context.request_stop()

          format_str = ('>> validation loss = %.2f')
          print (format_str % loss_val_value)

    class _TestScoreHook(tf.train.SessionRunHook):
      """Test Score Hook"""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        # return tf.train.SessionRunArgs([acc, prec, rec, f1])  # Asks for loss value.
        return tf.train.SessionRunArgs([loss_tst, tp, tn, fp, fn, acc, prec, rec, f1])

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          print('>> test scores: loss = %.3f, <%d,%d,%d,%d> acc = %.3f, prec = %.3f, rec = %.3f, f1 = %.3f' 
              % tuple(run_values.results))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss_trn),
               _LoggerHook(),
               _EarlyStoppingHook(),
               _TestScoreHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)

        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
    
        # mon_sess.run(train_op, options=run_options, run_metadata=run_metadata)
    
        # # Create the Timeline object, and write it to a json
        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('timeline.json', 'w') as f:
        #   f.write(ctf)

        
        # print(mon_sess.run(signals_trn)[0, 0, :])
        # print(mon_sess.run(signals_trn)[0, :])


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  # parse input parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=int, default=0,
                      help='0/1/2')

  parser.add_argument('recipe_no', type=int, default=1,
                      help='recipe no')
  parser.add_argument('step_no', type=int, default=1,
                      help='step no')
  parser.add_argument('device_id', type=str, default='PM6',
                      help='device id')
  args, unparsed = parser.parse_known_args()

  # Input filename
  FLAGS.data_fname = './{}-{}-{}.p'.format(
    args.recipe_no, args.step_no, args.device_id)

  # Directory where to write event logs and checkpoint.
  FLAGS.train_dir = './tf_logs/train'
  # Number of batches to run.
  FLAGS.max_steps = 5000
  # Whether to log device placement.
  FLAGS.log_device_placement = False
  # How often to log results to the console.
  FLAGS.log_frequency = 100
  
  tf.app.run()
