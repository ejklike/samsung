import tensorflow as tf
from tensorflow.contrib import layers
# import numpy as np

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var

def weighted_xe(pred, _y, minor_weight=1):
    weight_vector = 1 + _y[:, 1] * (minor_weight - 1)
    return tf.losses.softmax_cross_entropy(
        onehot_labels=_y,
        logits=pred,
        weights=weight_vector,
        label_smoothing=0,
        scope='weighted_xe',
        loss_collection=tf.GraphKeys.LOSSES,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    )

def conv_func(x, window_size, sensor_size, n_filter_in, n_filter_out, name='conv'):
    with tf.variable_scope(name) as scope:
        kernel = _variable_on_cpu(
            'weights', [window_size, sensor_size, n_filter_in, n_filter_out], 
            layers.xavier_initializer())
        biases = _variable_on_cpu(
            'biases', [n_filter_out], tf.zeros_initializer())
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv_out = tf.nn.relu(pre_activation, name=scope.name)
    return conv_out, kernel


def build_graph(x, keeprate, param_dict, FLAGS):
    time_size = FLAGS.time_size
    sensor_size = FLAGS.sensor_size

    n_conv = param_dict.n_conv
    window_size = param_dict.window_size
    n_filter = param_dict.n_filter
    pooling_size = param_dict.pooling_size

    n_hidden_list = param_dict.n_hidden_list
    pooling_stride = param_dict.pooling_stride

    # [batch_size, time_size, sensor_size, channels]
    with tf.device('/cpu:0'):
        x_reshaped = tf.reshape(x, [-1, time_size, sensor_size, 1], name='reshaped_x')

    # conv1
    conv, conv_kernel = conv_func(x_reshaped, 
        window_size, sensor_size, 1, n_filter, name='conv1')
    # pool1
    pool = tf.nn.max_pool(conv, 
        ksize=[1, pooling_size, 1, 1], strides=[1, pooling_stride, 1, 1], 
        padding='VALID', name='pool1')

    # the remained conv & pool
    for i in range(2, n_conv + 1):
        # conv i
        conv, _ = conv_func(pool, window_size//2 + 1, 1, n_filter, n_filter, name='conv%d'%i)
        # pool i
        this_pooling_stride = max(pooling_stride//2, 1)
        pool = tf.nn.max_pool(conv, 
            ksize=[1, pooling_size//2, 1, 1], 
            strides=[1, this_pooling_stride, 1, 1], 
            padding='VALID', name='pool%d'%i)

    feature = layers.flatten(pool, scope='flatten')
    for i, n_hidden in enumerate(n_hidden_list):
        feature = layers.fully_connected(feature, n_hidden, scope='fully_conn%d'%i)

    pred = layers.linear(feature, 2, scope='final_node')

    return pred, conv_kernel


def fdc_cnn(x, y, keeprate, param_dict, FLAGS):
    # build graph
    pred, _ = build_graph(x, keeprate, param_dict, FLAGS)

    # loss
    cost = weighted_xe(pred, y, minor_weight=FLAGS.minor_weight)

    return pred, cost


def group_cnn(x, y, keeprate, param_dict, FLAGS):
    # build graph
    pred, kernel = build_graph(x, keeprate, param_dict, FLAGS)

    # loss
    # [window_size, sensor_size, 1, n_filter] 
    # ==> squeeze [window_size, sensor_size, n_filter]
    # ==> unstack [window_size, sensor_size] * n_filter
    # ==> reduce_sum [sensor_size] * n_filter
    # ==> reduce_sum n_filter
    with tf.device('/cpu:0'):
        squeezed_kernel = tf.squeeze(kernel)
        # print(squeezed_kernel.get_shape())

        kernel_list = tf.unstack(squeezed_kernel, param_dict.n_filter, 2)
        # print(kernel_list[0].get_shape())

        kernel_list = [tf.sqrt(tf.reduce_sum(k**2, axis=0)) for k in kernel_list]
        # print(kernel_list[0].get_shape())
        
        sqrt_window_size = tf.sqrt(tf.constant(param_dict.window_size, dtype=tf.float32))
        kernel_list = [sqrt_window_size * tf.reduce_sum(k) for k in kernel_list]
        # print(kernel_list[0].get_shape())

        group_penalty = tf.reduce_sum(kernel_list) # sum by filter
    
    cost = weighted_xe(pred, y, minor_weight=FLAGS.minor_weight)
    cost += param_dict.group_c * group_penalty

    return pred, cost


# regularized group by sensor for all filters
def group_cnn2(x, y, keeprate, param_dict, FLAGS):
    # build graph
    pred, kernel = build_graph(x, keeprate, param_dict, FLAGS)
    # loss
    # [window_size, sensor_size, 1, n_filter] 
    # ==> squeeze [window_size, sensor_size, n_filter]
    # ==> unstack [window_size, sensor_size] * n_filter
    # ==> reduce_sum [sensor_size] * n_filter
    # ==> reduce_sum n_filter
    with tf.device('/cpu:0'):
        squeezed_kernel = tf.squeeze(kernel)
        # print(squeezed_kernel.get_shape())

        kernel_list = tf.unstack(squeezed_kernel, param_dict.n_filter, 2)
        # print(kernel_list[0].get_shape())

        unified_kernel = tf.stack(kernel_list, axis=0)
        l2_norm_list = tf.sqrt(tf.reduce_sum(unified_kernel**2, axis=0))

        sqrt_window_kernel_size = tf.sqrt(
            tf.constant(param_dict.window_size * param_dict.n_filter, dtype=tf.float32))
        group_penalty = sqrt_window_kernel_size * tf.reduce_sum(l2_norm_list)

    cost = weighted_xe(pred, y, minor_weight=FLAGS.minor_weight)
    cost += param_dict.group_c * group_penalty

    return pred, cost










def fdc_cnn_(x, y, keeprate, param_dict, FLAGS):
    time_size = FLAGS.time_size
    sensor_size = FLAGS.sensor_size

    batch_size = FLAGS.batch_size

    window_size = param_dict.window_size
    n_filter = param_dict.n_filter
    pooling_size = param_dict.pooling_size
    n_hidden = param_dict.n_hidden

    # [batch_size, time_size, sensor_size, channels]
    with tf.device('/cpu:0'):
        x_reshaped = tf.reshape(x, [-1, time_size, sensor_size, 1], name='reshaped_x')

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_on_cpu(
            'weights', [window_size, sensor_size, n_filter, n_filter], 
            layers.xavier_initializer())

        biases = _variable_on_cpu(
            'biases', [n_filter], tf.zeros_initializer())
        conv = tf.nn.conv2d(x_reshaped, kernel, [1, 1, 1, 1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # conv1 = tf.nn.bias_add(conv, biases)

    # pool1
    pool1 = tf.nn.max_pool(conv1, 
        ksize=[1, pooling_size, 1, 1], strides=[1, 2, 1, 1],
        padding='VALID', name='pool1')

    # # conv2
    # with tf.variable_scope('conv2') as scope:
    #     kernel = _variable_on_cpu(
    #         'weights', [window_size//2, 1, n_filter, n_filter], 
    #         layers.xavier_initializer())
    #     biases = _variable_on_cpu(
    #         'biases', [n_filter], tf.zeros_initializer())
    #     conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # # pool2
    # pool2 = tf.nn.max_pool(conv2, 
    #     ksize=[1, pooling_size, 1, 1], strides=[1, 2, 1, 1],
    #     padding='VALID', name='pool1')

    # flatten layer
    flat = layers.flatten(pool1, scope='flatten')
    # flat = layers.flatten(pool2, scope='flatten')
    fully_conn = layers.fully_connected(flat, n_hidden, scope='fully_conn1')
    fully_conn = layers.fully_connected(fully_conn, n_hidden//2, scope='fully_conn2')

    pred = layers.linear(fully_conn, 2, scope='final_node')

    # loss
    cost = weighted_xe(pred, y, minor_weight=FLAGS.minor_weight)

    return pred, cost



def group_cnn_(x, y, keeprate, param_dict, FLAGS):
    time_size = FLAGS.time_size
    sensor_size = FLAGS.sensor_size

    batch_size = FLAGS.batch_size
    
    group_c = param_dict.group_c
    
    window_size = param_dict.window_size
    n_filter = param_dict.n_filter
    pooling_size = param_dict.pooling_size
    n_hidden = param_dict.n_hidden

    # [batch_size, time_size, sensor_size, channels]
    with tf.device('/cpu:0'):
        x_reshaped = tf.reshape(x, [-1, time_size, sensor_size, 1], name='reshaped_x')

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_on_cpu(
            'weights', [window_size, sensor_size, 1, n_filter], 
            layers.xavier_initializer())
        biases = _variable_on_cpu(
            'biases', [n_filter], tf.zeros_initializer())
        conv = tf.nn.conv2d(x_reshaped, kernel, [1, 1, 1, 1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, 
        ksize=[1, pooling_size, 1, 1], strides=[1, 2, 1, 1],
        padding='VALID', name='pool1')

    # # conv2
    # with tf.variable_scope('conv2') as scope:
    #     kernel = _variable_on_cpu(
    #         'weights', [window_size//2, 1, n_filter, n_filter], 
    #         layers.xavier_initializer())
    #     biases = _variable_on_cpu(
    #         'biases', [n_filter], tf.zeros_initializer())
    #     conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # # pool2
    # pool2 = tf.nn.max_pool(conv2, 
    #     ksize=[1, pooling_size, 1, 1], strides=[1, 2, 1, 1],
    #     padding='VALID', name='pool1')

    # flatten layer
    flat = layers.flatten(pool1, scope='flatten')
    # flat = layers.flatten(pool2, scope='flatten')
    fully_conn = layers.fully_connected(flat, n_hidden, scope='fully_conn1')
    fully_conn = layers.fully_connected(fully_conn, n_hidden//2, scope='fully_conn2')

    pred = layers.linear(fully_conn, 2, scope='final_node')

    # loss
    # [window_size, sensor_size, 1, n_filter] 
    # ==> squeeze [window_size, sensor_size, n_filter]
    # ==> unstack [window_size, sensor_size] * n_filter
    # ==> reduce_sum [sensor_size] * n_filter
    # ==> reduce_sum n_filter
    with tf.device('/cpu:0'):
        squeezed_kernel = tf.squeeze(kernel)
        # print(squeezed_kernel.get_shape())

        kernel_list = tf.unstack(squeezed_kernel, n_filter, 2)
        # print(kernel_list[0].get_shape())

        kernel_list = [tf.sqrt(tf.reduce_sum(k**2, axis=0)) for k in kernel_list]
        # print(kernel_list[0].get_shape())
        
        sqrt_window_size = tf.sqrt(tf.constant(window_size, dtype=tf.float32))
        kernel_list = [sqrt_window_size * tf.reduce_sum(k) for k in kernel_list]
        # print(kernel_list[0].get_shape())

        group_penalty = tf.reduce_sum(kernel_list) # sum by filter
    
    cost = weighted_xe(pred, y, minor_weight=FLAGS.minor_weight)
    cost += group_c * group_penalty

    return pred, cost




# regularized group by sensor for all filters
def group_cnn2_(x, y, keeprate, param_dict, FLAGS):
    time_size = FLAGS.time_size
    sensor_size = FLAGS.sensor_size

    batch_size = FLAGS.batch_size
    
    group_c = param_dict.group_c
    
    window_size = param_dict.window_size
    n_filter = param_dict.n_filter
    pooling_size = param_dict.pooling_size
    n_hidden = param_dict.n_hidden

    # [batch_size, time_size, sensor_size, channels]
    with tf.device('/cpu:0'):
        x_reshaped = tf.reshape(x, [-1, time_size, sensor_size, 1], name='reshaped_x')

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_on_cpu(
            'weights', [window_size, sensor_size, 1, n_filter], 
            layers.xavier_initializer())
        biases = _variable_on_cpu(
            'biases', [n_filter], tf.zeros_initializer())
        conv = tf.nn.conv2d(x_reshaped, kernel, [1, 1, 1, 1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, 
        ksize=[1, pooling_size, 1, 1], strides=[1, 2, 1, 1],
        padding='VALID', name='pool1')

    # # conv2
    # with tf.variable_scope('conv2') as scope:
    #     kernel = _variable_on_cpu(
    #         'weights', [window_size//2, 1, n_filter, n_filter], 
    #         layers.xavier_initializer())
    #     biases = _variable_on_cpu(
    #         'biases', [n_filter], tf.zeros_initializer())
    #     conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # # pool2
    # pool2 = tf.nn.max_pool(conv2, 
    #     ksize=[1, pooling_size, 1, 1], strides=[1, 2, 1, 1],
    #     padding='VALID', name='pool1')

    # flatten layer
    flat = layers.flatten(pool1, scope='flatten')
    # flat = layers.flatten(pool2, scope='flatten')
    fully_conn = layers.fully_connected(flat, n_hidden, scope='fully_conn1')
    fully_conn = layers.fully_connected(fully_conn, n_hidden//2, scope='fully_conn2')

    pred = layers.linear(fully_conn, 2, scope='final_node')

    # loss
    # [window_size, sensor_size, 1, n_filter] 
    # ==> squeeze [window_size, sensor_size, n_filter]
    # ==> unstack [window_size, sensor_size] * n_filter
    # ==> reduce_sum [sensor_size] * n_filter
    # ==> reduce_sum n_filter
    with tf.device('/cpu:0'):
        squeezed_kernel = tf.squeeze(kernel)
        # print(squeezed_kernel.get_shape())

        kernel_list = tf.unstack(squeezed_kernel, n_filter, 2)
        # print(kernel_list[0].get_shape())

        unified_kernel = tf.stack(kernel_list, axis=0)
        l2_norm_list = tf.sqrt(tf.reduce_sum(unified_kernel**2, axis=0))
        
        sqrt_window_kernel_size = tf.sqrt(
            tf.constant(window_size * n_filter, dtype=tf.float32))
        group_penalty = sqrt_window_kernel_size * tf.reduce_sum(l2_norm_list)

    cost = weighted_xe(pred, y, minor_weight=FLAGS.minor_weight)
    cost += group_c * group_penalty

    return pred, cost


# lasso no group
def group_cnn3(x, y, keeprate, param_dict, FLAGS):
    time_size = FLAGS.time_size
    sensor_size = FLAGS.sensor_size

    batch_size = FLAGS.batch_size
    
    group_c = param_dict.group_c
    
    window_size = param_dict.window_size
    n_filter = param_dict.n_filter
    pooling_size = param_dict.pooling_size
    n_hidden = param_dict.n_hidden

    # [batch_size, time_size, sensor_size, channels]
    with tf.device('/cpu:0'):
        x_reshaped = tf.reshape(x, [-1, time_size, sensor_size, 1], name='reshaped_x')

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_on_cpu(
            'weights', [window_size, sensor_size, 1, n_filter], 
            layers.xavier_initializer())
        biases = _variable_on_cpu(
            'biases', [n_filter], tf.zeros_initializer())
        conv = tf.nn.conv2d(x_reshaped, kernel, [1, 1, 1, 1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, 
        ksize=[1, pooling_size, 1, 1], strides=[1, 2, 1, 1],
        padding='VALID', name='pool1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_on_cpu(
            'weights', [window_size//2, 1, n_filter, n_filter], 
            layers.xavier_initializer())
        biases = _variable_on_cpu(
            'biases', [n_filter], tf.zeros_initializer())
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # pool2
    pool2 = tf.nn.max_pool(conv2, 
        ksize=[1, pooling_size, 1, 1], strides=[1, 2, 1, 1],
        padding='VALID', name='pool1')

    # flatten layer
    flat = layers.flatten(pool2, scope='flatten')
    fully_conn1 = layers.fully_connected(flat, n_hidden, scope='fully_conn1')
    fully_conn2 = layers.fully_connected(fully_conn1, n_hidden//2, scope='fully_conn2')

    pred = layers.linear(fully_conn2, 2, scope='final_node')

    # loss
    # [window_size, sensor_size, 1, n_filter] 
    # ==> squeeze [window_size, sensor_size, n_filter]
    # ==> unstack [window_size, sensor_size] * n_filter
    # ==> reduce_sum [sensor_size] * n_filter
    # ==> reduce_sum n_filter
    with tf.device('/cpu:0'):
        squeezed_kernel = tf.squeeze(kernel)
        # print(squeezed_kernel.get_shape())

        kernel_list = tf.unstack(squeezed_kernel, n_filter, 2)
        # print(kernel_list[0].get_shape())

        unified_kernel = tf.stack(kernel_list, axis=0)
        l1_norms = tf.sqrt(unified_kernel**2)
        
        sqrt_window_kernel_size = tf.sqrt(
            tf.constant(window_size * n_filter * sensor_size, dtype=tf.float32))
        group_penalty = sqrt_window_kernel_size * tf.reduce_sum(l1_norms)

    cost = weighted_xe(pred, y, minor_weight=FLAGS.minor_weight)
    cost += group_c * group_penalty

    return pred, cost




def group_cnn_multi_windows(x, y, keeprate, param_dict, FLAGS):
    time_size = FLAGS.time_size
    sensor_size = FLAGS.sensor_size

    batch_size = FLAGS.batch_size
    
    group_c = param_dict.group_c
    
    window_size = param_dict.window_size
    n_filter = param_dict.n_filter
    pooling_size = param_dict.pooling_size
    n_hidden = param_dict.n_hidden

    # [batch_size, time_size, sensor_size, channels]
    with tf.device('/cpu:0'):
        x_reshaped = tf.reshape(x, [-1, time_size, sensor_size, 1], name='reshaped_x')

    def conv_func(x, window_size, sensor_size, n_filter):
        kernel = _variable_on_cpu(
            'weights', [window_size, sensor_size, 1, n_filter], 
            layers.xavier_initializer())
        biases = _variable_on_cpu(
            'biases', [n_filter], tf.zeros_initializer())
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv_out = tf.nn.relu(pre_activation, name=scope.name)
        return conv_out, kernel

    window_size_list = [1, 3, 5]
    n_filter_list = [1, 2, 7]

    kernel_list, flat_list = [], []
    for window_size, n_filter in zip(window_size_list, n_filter_list):
        # conv1
        with tf.variable_scope('conv%d'%window_size) as scope:
            conv, kernel = conv_func(x_reshaped, window_size, sensor_size, n_filter)
        # pool1
        pool = tf.nn.max_pool(conv, 
            ksize=[1, pooling_size, 1, 1], strides=[1, 2, 1, 1],
            padding='VALID', name='pool%d'%window_size)
        flat = layers.flatten(pool, scope='flatten%d'%window_size)
        
        kernel_list.append(kernel)
        flat_list.append(flat)

    # flatten layer
    concat = tf.concat(flat_list, 1, name='concat')
    concat_drop = tf.nn.dropout(concat, keeprate)
    fully_conn1 = layers.fully_connected(concat_drop, n_hidden, scope='fully_conn1')
    fully_conn1_drop = tf.nn.dropout(fully_conn1, keeprate)
    fully_conn2 = layers.fully_connected(fully_conn1, n_hidden//2, scope='fully_conn2')

    pred = layers.linear(fully_conn2, 2, scope='final_node')


    # [window_size, sensor_size, 1, n_filter] 
    # ==> squeeze [window_size, sensor_size, n_filter]
    # ==> unstack [window_size, sensor_size] * n_filter
    # ==> reduce_sum [sensor_size] * n_filter
    # ==> reduce_sum n_filter
    def reg_term(kernel, group_c, window_size, n_filter):
        with tf.device('/cpu:0'):
            # print(kernel.get_shape())

            squeezed_kernel = tf.squeeze(kernel, [2])
            # print(squeezed_kernel.get_shape())

            kernel_list = tf.unstack(squeezed_kernel, n_filter, 2)
            # print(kernel_list[0].get_shape())

            kernel_list = [tf.sqrt(tf.reduce_sum(k**2, axis=0)) for k in kernel_list]
            # print(kernel_list[0].get_shape())
            
            sqrt_window_size = tf.sqrt(tf.constant(window_size, dtype=tf.float32))
            kernel_list = [tf.reduce_sum(k) for k in kernel_list]
            # print(kernel_list[0].get_shape())

            group_penalty = sqrt_window_size * tf.reduce_sum(kernel_list) # sum by filter
            # print(group_penalty.get_shape())

        # group_c = tf.constant(group_c, dtype=tf.float32)
        return group_c * group_penalty

    # loss
    cost = weighted_xe(pred, y, minor_weight=FLAGS.minor_weight)
    for window_size, n_filter, kernel in zip(window_size_list, n_filter_list, kernel_list):
        cost += reg_term(kernel, group_c, window_size, n_filter)

    return pred, cost

    # # local2
    # with tf.variable_scope('local2') as scope:
    #     # Move everything into depth so we can perform a single matrix multiply.
    #     reshape = tf.reshape(pool1, [batch_size, -1])
    #     dim = reshape.get_shape()[1].value
    #     weights = _variable_on_cpu(
    #         'weights', [dim, n_hidden], tf.truncated_normal_initializer(stddev=5e-2))
    #     biases = _variable_on_cpu(
    #         'biases', [n_hidden], tf.constant_initializer(0.1))
    #     local2 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    #  # linear layer(WX + b),
    # # We don't apply softmax here because
    # # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # # and performs the softmax internally for efficiency.
    # with tf.variable_scope('softmax_linear') as scope:
    #     weights = _variable_on_cpu(
    #         'weights', [n_hidden, 2], tf.truncated_normal_initializer(stddev=5e-2))
    #     biases = _variable_on_cpu(
    #         'biases', [2], tf.constant_initializer(0.0))
    #     softmax_linear = tf.add(tf.matmul(local2, weights), biases, name=scope.name)

    # return softmax_linear