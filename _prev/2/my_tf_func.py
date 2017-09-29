import tensorflow as tf

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












def get_data()
  raw_data = pickle.load(open(data_fname, 'rb'))
  signals, labels = raw_data['signals'], raw_data['labels']

  raw_data = pickle.load(open('./PM2.p', 'rb'))
  signals2, labels2 = raw_data['signals'], raw_data['labels']

  raw_data = pickle.load(open('./PM6.p', 'rb'))
  signals3, labels3 = raw_data['signals'], raw_data['labels']

  signals = np.concatenate([signals, signals2, signals3], axis=0)
  labels = np.concatenate([labels, labels2, labels3], axis=0)


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




  raw_data = pickle.load(open(data_fname, 'rb'))
  signals = raw_data['signals']
  labels_loc, labels_ges = raw_data['labels_locomotion'], raw_data['labels_gesture']

  labels = labels_loc[:, 0:2]