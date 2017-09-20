import argparse
from datetime import datetime
import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from tensorflow.python.client import timeline

import my_graph

FLAGS = tf.app.flags.FLAGS

def save_results(iter, loss, acc, prec, rec, f1):
  # record file existence check
  if not os.path.exists(FLAGS.record_fname):
    with open(FLAGS.record_fname, 'w') as fout:
      fout.write('datetime,data_fname,reg_type,wd,iter,loss,acc,prec,rec,f1')
      fout.write('\n')

  with open(FLAGS.record_fname, 'a') as fout:
    fout.write('{},'.format(FLAGS.timestamp))
    fout.write('{},{},{},'.format(FLAGS.data_fname, FLAGS.reg_type, FLAGS.wd))
    fout.write('{},{},{},{},{},{},'.format(iter, loss, acc, prec, rec, f1))
    fout.write('\n')


def train():
  """Train a model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get signals and labels
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      signals_trn, labels_trn= my_graph.inputs(
        FLAGS.data_fname, train=True, val=False)
      signals_tst, labels_tst = my_graph.inputs(
        FLAGS.data_fname, train=False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    # window_size = 5
    # n_filter = 10
    # n_hidden = 30

    with tf.variable_scope('inference') as scope:
      logits_trn = my_graph.inference(signals_trn, reg_type=FLAGS.reg_type, wd=FLAGS.wd)
      scope.reuse_variables()
      logits_tst = my_graph.inference(signals_tst, reg_type=FLAGS.reg_type, wd=FLAGS.wd)

    # Calculate loss.
    loss_trn = my_graph.loss(logits_trn, labels_trn)
    loss_tst, test_metric = my_graph.loss(logits_tst, labels_tst, test_metric=True)

    # test scores
    acc, prec, rec, f1 = test_metric

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = my_graph.train(loss_trn, global_step)

    # class _StopAtStepHook(tf.train.StopAtStepHook):

    #   def after_run(self, run_context, run_values):
    #     global_step = run_values.results
    #     # if global_step >= self._last_step:
    #     if global_step > 0 and global_step % self._last_step == 0:
    #       run_context.request_stop()
    #       print('---', global_step)

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


    class _TestScoreHook(tf.train.SessionRunHook):
      """Test Score Hook"""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        # return tf.train.SessionRunArgs([acc, prec, rec, f1])  # Asks for loss value.
        return tf.train.SessionRunArgs([loss_tst, acc, prec, rec, f1])

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          test_msg = (
            '>> test scores; loss, acc, prec, rec, f1 = '
            '%.3f, %.3f, %.3f, %.3f, %.3f'
            ) % tuple(run_values.results)
          print(test_msg)
          self._run_values = run_values.results

      def end(self, session):
        save_results(self._step, *self._run_values)


    # training session
    print('-'*100)
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
              tf.train.NanTensorHook(loss_trn),
              _LoggerHook(),
              _TestScoreHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=FLAGS.gpu_usage_option)) as mon_sess:
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
        
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  # parse input parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('recipe_no', type=int, default=1,
                      help='recipe no')
  parser.add_argument('step_no', type=int, default=1,
                      help='step no')
  parser.add_argument('device_id', type=int, default=1,
                      help='device id')

  parser.add_argument('--gpu_no', type=int, default=1,
                      help='gpu device number')
  parser.add_argument('--gpu_usage', type=float, default=0.45,
                      help='gpu usage')
  parser.add_argument('--model', type=int, default=0,
                      help='0/1/2/3')
  parser.add_argument('--wd', type=float, default=0,
                      help='weight decaying factor')

  args, unparsed = parser.parse_known_args()

  # Input filename
  FLAGS.data_fname = './{}-{}-PM{}.p'.format(
    args.recipe_no, args.step_no, args.device_id)
  FLAGS.record_fname = './{}-{}-PM{}.csv'.format(
    args.recipe_no, args.step_no, args.device_id)
  # Model Type
  FLAGS.reg_type = {
    0: 'vanilla', 1: 'l1', 2: 'group1', 3: 'group2'
  }[args.model]
  # Weight Decaying constant
  FLAGS.wd = None if args.model == 0 else args.wd
  # GPU setting
  os.environ['CUDA_VISIBLE_DEVICES'] = '%d' %args.gpu_no
  # Assume that you have 12GB of GPU memory and want to allocate ~4GB: 0.333
  FLAGS.gpu_usage_option = tf.GPUOptions(
            per_process_gpu_memory_fraction=args.gpu_usage)

  # Directory where to write event logs and checkpoint.
  FLAGS.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  FLAGS.train_dir = './tf_logs/train_{}_{}'.format(FLAGS.reg_type, FLAGS.timestamp)
  # Number of batches to run.
  FLAGS.max_steps = 3000
  # Whether to log device placement.
  FLAGS.log_device_placement = False
  # How often to log results to the console.
  FLAGS.log_frequency = 100
  
  tf.app.run()