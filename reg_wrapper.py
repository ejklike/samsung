import numbers

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops

def group_regularizer(scale, scope=None, unified=False):
  """Returns a function that can be used to apply group regularization to weights.
  """
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
      kernel_list = tf.unstack(value=weights, num=n_filter, axis=2) # unpack by kernels
      axis = 0 if unified else 1 ## 
      unified_kernel = tf.concat(kernel_list, axis=axis)
    
    with ops.name_scope(scope, 'group_regularizer', [unified_kernel]) as name:
      my_scale = ops.convert_to_tensor(scale, ##
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      weight_regularization = tf.reduce_sum(
        tf.sqrt(tf.reduce_sum(unified_kernel**2, axis=0)))
      return standard_ops.multiply(my_scale, weight_regularization, name=name)
  
  return group_l1