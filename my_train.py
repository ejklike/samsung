import argparse
from datetime import datetime
import time

import tensorflow as tf

import my_graph

FLAGS = tf.app.flags.FLAGS

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      signals, labels = my_graph.inputs(FLAGS.data_fname, train=True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = my_graph.inference(signals)

    # Calculate loss.
    loss = my_graph.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = my_graph.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

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

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
        # print(mon_sess.run(signals)[0, 0, :])


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
  FLAGS.max_steps = 3000
  # Whether to log device placement.
  FLAGS.log_device_placement = False
  # How often to log results to the console.
  FLAGS.log_frequency = 100
  
  tf.app.run()
