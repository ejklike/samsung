
import argparse
import os
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import my_graph

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './tf_logs/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './tf_logs/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 10,
                            """How often to run the eval.""")
# tf.app.flags.DEFINE_integer('num_examples', 10000,
#                             """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def my_metrics(logits, labels):
    original_y, predicted_y = tf.argmax(labels, 1), tf.argmax(logits, 1)
    correct_pred = tf.equal(original_y, predicted_y)
    incorrect_pred = tf.logical_not(correct_pred)

    # confusion matrix
    #               true 1   true 0
    # predicted 1     TP       FP
    # predicted 0     FN       TN

    tp = tf.reduce_sum(tf.boolean_mask(original_y, correct_pred))
    tn = tf.reduce_sum(tf.boolean_mask(1 - original_y, correct_pred))
    fp = tf.reduce_sum(tf.boolean_mask(original_y, incorrect_pred))
    fn = tf.reduce_sum(tf.boolean_mask(1 - original_y, incorrect_pred))
    
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.int64))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 / (1/precision + 1/recall)
    
    return accuracy, precision, recall, fscore


def eval_once(saver, summary_writer, logits, labels, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

    #   num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    #   true_count = 0  # Counts the number of correct predictions.
    #   total_sample_count = num_iter * FLAGS.batch_size
    #   step = 0
    #   while step < num_iter and not coord.should_stop():
    #     predictions = sess.run([top_k_op])
    #     true_count += np.sum(predictions)
    #     step += 1

      # Compute precision @ 1.
    #   precision = true_count / total_sample_count
      
      acc, prec, rec, f1 = sess.run(my_metrics(logits, labels))

      print('%s: acc = %.3f, prec = %.3f, rec = %.3f, f1 = %.3f' 
            % (datetime.now(), acc, prec, rec, f1))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Accuracy', simple_value=acc)
      summary.value.add(tag='Precision', simple_value=prec)
      summary.value.add(tag='Recall', simple_value=rec)
      summary.value.add(tag='F1score', simple_value=f1)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(train=False):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
    # Get images and labels for CIFAR-10.
    signals, labels = my_graph.inputs(FLAGS.data_fname, train=False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    with tf.variable_scope('inference') as scope:
      logits = my_graph.inference(signals)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        my_graph.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, logits, labels, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate(train=False)


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

  tf.app.run()
