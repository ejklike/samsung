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

  if which_data == 'drill':
    raw_data = pickle.load(open(data_fname, 'rb'))
    signals = raw_data['signals']
    labels_loc = raw_data['labels_locomotion']
    labels_ges = raw_data['labels_gesture']
    labels = labels_loc[:, 0:2]
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
  return signals, labels