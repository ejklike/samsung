import pickle

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# # batch generation
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def load(which_data, train=True):
  if train:
    data_fname = './data/{}_train.p'.format(which_data)
  else:
    data_fname = './data/{}_test.p'.format(which_data)

  if which_data[:3] == 'opp':
    raw_data = pickle.load(open(data_fname, 'rb'))
    signals = raw_data['signals']
    labels_loc = raw_data['labels_locomotion']
    labels_ges = raw_data['labels_gesture']
    
    # gesture label order
    #      0, 504605, 504608, 504611, 504616, 504617, 504619, 504620,
    # 505606, 506605, 506608, 506611, 506616, 506617, 506619, 506620,
    # 507621, 508612
    # locomotion label order
    # 0, 101, 102, 104, 105

    label_idx = 1
    labels_one_col = labels_loc[:, label_idx]
    def binary_to_dummies(arr):
        arr = arr.reshape(-1, 1)
        return np.concatenate((1-arr, arr), axis=1)
    labels = binary_to_dummies(labels_one_col)
    class_count = np.sum(labels, axis=0)
    print('class imbalance: {}, {} ({:.03}%)'.format(
        *class_count, 
        class_count[1] / np.sum(class_count) * 100)
    )

  elif which_data == 'samsung':
    raw_data = pickle.load(open(data_fname, 'rb'))
    signals = raw_data['signals']
    labels = raw_data['labels']

  else:
    raise Exception('Existing no data matching with your input str.')
  
  print(signals.shape, labels.shape)

  signals = signals.astype(np.float32)
  labels = labels.astype(np.float32)

  return signals, labels