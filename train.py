import argparse
from datetime import datetime
import time
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

import dataloader
import graph

FLAGS = tf.app.flags.FLAGS
FLAGS.dtype = tf.float32
FLAGS.num_threads = 10

def train():
  """Train a model for a number of steps."""

  # Get signals and labels
  signals_trn, labels_trn = dataloader.load(which_data=FLAGS.data_type, train=True)
  

  signals_trn, signals_val, labels_trn, labels_val = train_test_split(
      signals_trn, labels_trn, test_size=0.2, random_state=42, stratify=labels_trn[:, 1])

  signals_tst, labels_tst = dataloader.load(which_data=FLAGS.data_type, train=False)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Generate batches
    signals_trn, labels_trn = graph.inputs(signals_trn, labels_trn)
    signals_val, labels_val = graph.inputs(signals_val, labels_val)
    signals_tst, labels_tst = graph.inputs(signals_tst, labels_tst)

    # Build inference graph
    with tf.variable_scope('inference') as scope:
      logits_trn = graph.inference(signals_trn)
      scope.reuse_variables()
      logits_val = graph.inference(signals_val)
      scope.reuse_variables()
      logits_tst = graph.inference(signals_tst)

    # Calculate loss.
    loss_trn, xe_loss, reg_loss = graph.loss(logits_trn, labels_trn)
    loss_val, _, _ = graph.loss(logits_val, labels_val)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = graph.train(loss_trn, global_step)

    # Add the Op to compare the logits to the labels during evaluation.
    acc, prec, rec, f1 = graph.evaluation(logits_tst, labels_tst)

    # training session
    print('-'*50)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create the op for initializing variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    sess.run(init_op)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # And then after everything is built, start the training loop.
    try:
      step = 0
      prev_loss_val_value = 1e10
      tolerance_count, max_tolerance_count = 0, 3 * FLAGS.log_frequency
      while not coord.should_stop():
        start_time = time.time()

        # Run one step of the model.
        _, loss_value = sess.run([train_op, loss_trn])

        duration = time.time() - start_time

        # Validation to determine early stopping.
        loss_val_value = sess.run(loss_val)
        if loss_val_value > prev_loss_val_value:
          tolerance_count += 1
        else:
          tolerance_count = 0
        if tolerance_count >= max_tolerance_count:
          coord.request_stop()
        prev_loss_val_value = loss_val_value

        # Write the summaries and print an overview fairly often.
        if step % FLAGS.log_frequency == 0:
          examples_per_sec = FLAGS.batch_size / duration

          # Print status to stdout.
          print('Step %d: trn_loss = %.2f (%.2f + %.2f), '
                'val_loss = %.2f, (%.3f sec/batch; %.1f examples/sec)' 
                % (step, loss_value, *sess.run([xe_loss, reg_loss]), 
                   loss_val_value, duration, examples_per_sec))

          # Update the events file.
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)

        # Save a checkpoint periodically.
        if step % (3 * FLAGS.log_frequency) == 0:
          # print('Saving')
          # saver.save(sess, FLAGS.train_dir, global_step=step)
          test_values = sess.run([acc, prec, rec, f1])
          print('>> Test score:')
          print('>>', test_values)
          # print('>> Test score: {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(test_values))

        step += 1
    except tf.errors.OutOfRangeError:
      # print('Saving')
      # saver.save(sess, FLAGS.train_dir, global_step=step)
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    except KeyboardInterrupt:
      print('>> Traing stopped by user!')
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  # parse input parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('data_type', type=str, default='drill',
                      help='data: drill/samsung')

  parser.add_argument('--gpu_no', type=str, default='-1',
                      help='gpu device number')
  # parser.add_argument('--gpu_usage', type=float, default=0.45,
  #                     help='gpu usage')
  parser.add_argument('--model', type=int, default=0,
                      help='0/1/2/3')
  parser.add_argument('--wd', type=float, default=0,
                      help='weight decaying factor')

  parser.add_argument('--lr', type=float, default=0.0001,
                      help='initial learning rate')
  parser.add_argument('--log_freq', type=int, default=100,
                      help='log frequency')

  args, unparsed = parser.parse_known_args()

  # Data Type
  FLAGS.data_type = args.data_type
  if FLAGS.data_type == 'drill':
    FLAGS.batch_size = 100
    FLAGS.num_epochs = 1
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1' if args.gpu_no == '-1' else args.gpu_no
  elif FLAGS.data_type == 'samsung':
    FLAGS.batch_size = 100
    FLAGS.num_epochs = 1
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0' if args.gpu_no == '-1' else args.gpu_no
  
  # GPU setting
  os.environ['CUDA_VISIBLE_DEVICES'] = '%d' %args.model
  # os.environ['CUDA_VISIBLE_DEVICES'] = '%d' %args.gpu_no
  # Assume that you have 12GB of GPU memory and want to allocate ~4GB: 0.333
  # FLAGS.gpu_usage_option = tf.GPUOptions(allow_growth=True)
            # per_process_gpu_memory_fraction=args.gpu_usage)

  # Model Type
  FLAGS.reg_type = {
    0: 'vanilla', 1: 'l1', 2: 'group1', 3: 'group2'
  }[args.model]
  # Weight Decaying constant
  FLAGS.wd = None if args.model == 0 else args.wd
  
  FLAGS.lr = args.lr
  FLAGS.log_frequency = args.log_freq



  

  # Directory where to write event logs and checkpoint.
  FLAGS.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  FLAGS.train_dir = './tf_logs/{}/train_{}_{}/{}'.format(
      FLAGS.data_type, FLAGS.reg_type, FLAGS.wd, FLAGS.timestamp)
  
  
  
  # # Number of batches to run.
  # FLAGS.max_steps = 3000
  # # Whether to log device placement.
  # FLAGS.log_device_placement = False
  
  tf.app.run()