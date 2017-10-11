import os
from datetime import datetime
import time

import numpy as np
import tensorflow as tf

from tf_func import my_metrics

log_root = 'tf_logs'
# tf.gfile.DeleteRecursively(logroot)

def make_get_path(*args):
    current_path = './'
    for arg in args:
        current_path = os.path.join(current_path, arg)
        if not os.path.exists(current_path):
            os.makedirs(current_path)
    return current_path


# class BatchGenerator(object):
#     def __init__(self, X_trn, y_trn, FLAGS):
#         self.batch_size = FLAGS.batch_size
#         self.max_iter = FLAGS.max_iter
#         self.X_trn, self.y_trn = X_trn, y_trn
#         self.normal_index = np.where(y_trn[:,1]==0)[0]
#         self.fault_index = np.where(y_trn[:,1]==1)[0]

#         fault_ratio = np.sum(y_trn[:, 1]) / y_trn.shape[0]
#         self.n_fault_in_batch = int(self.batch_size * fault_ratio)
#         self.n_normal_in_batch = self.batch_size - self.n_fault_in_batch

#     def next_batch(self):
#         this_normal_index = np.random.choice(self.normal_index, self.n_normal_in_batch)
#         this_fault_index = np.random.choice(self.fault_index, self.n_fault_in_batch)
#         this_index = np.concatenate((this_normal_index, this_fault_index))
#         return self.X_trn[this_index, :], self.y_trn[this_index]


# def batch_generator(X_trn, y_trn, FLAGS):
#     normal_index = np.where(y_trn[:,1]==0)[0]
#     fault_index = np.where(y_trn[:,1]==1)[0]

#     fault_ratio = np.sum(y_trn[:, 1]) / y_trn.shape[0]
#     n_fault_in_batch = int(FLAGS.batch_size * fault_ratio)
#     n_normal_in_batch = FLAGS.batch_size - n_fault_in_batch

#     for _ in range(FLAGS.max_iter):
#         this_normal_index = np.random.choice(normal_index, n_normal_in_batch)
#         this_fault_index = np.random.choice(fault_index, n_fault_in_batch)
#         this_index = np.concatenate((this_normal_index, this_fault_index))
#         yield X_trn[this_index, :], y_trn[this_index, :]


class Model(object):
    def __init__(self, param_dict, FLAGS):
        # init graph
        tf.reset_default_graph()

        # input placeholders
        self._x = tf.placeholder(tf.float32, 
            [None, FLAGS.time_size, FLAGS.sensor_size], name='input_x')
        self._y = tf.placeholder(tf.float32, [None, 2], name='input_y')
        self._keeprate = tf.placeholder(tf.float32, name='keeprate')

        # build graph
        self.pred, self.cost = FLAGS.model_name(self._x, self._y, self._keeprate, param_dict, FLAGS)


    def train(self, X_trn, y_trn, X_val, y_val, X_tst, y_tst, FLAGS):
        _x, _y, _keeprate, pred, cost = self._x, self._y, self._keeprate, self.pred, self.cost

        start_time = time.time()

        # Define a Session
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=FLAGS.gpu_usage)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # self._x = tf.constant(X_trn)
        # self._y = tf.constant(y_trn)

        # self._y_val = tf.constant()

        # X_trn_initializer = tf.placeholder(dtype=X_trn.dtype, shape=X_trn.shape)
        # y_trn_initializer = tf.placeholder(dtype=y_trn.dtype, shape=y_trn.shape)
        # input_X_trn = tf.Variable(X_trn_initializer, trainable=False, collections=[])
        # input_y_trn = tf.Variable(y_trn_initializer, trainable=False, collections=[])

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate).minimize(cost)

        # Evaluate model
        accuracy, precision, recall, fscore = my_metrics(self._y, self.pred)

        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # sess.run(input_X_trn.initializer, feed_dict={X_trn_initializer: X_trn})
        # sess.run(input_y_trn.initializer, feed_dict={y_trn_initializer: y_trn})

        def select_data(what):
            data_set = dict(
                trn={_x: X_trn, _y: y_trn, _keeprate: FLAGS.keeprate},
                val={_x: X_val, _y: y_val, _keeprate: 1},
                tst={_x:X_tst, _y:y_tst, _keeprate: 1}
            )
            return data_set[what]

        loss_val_best = 1e10
        for i in range(FLAGS.max_iter):
            # Run optimization op (backprop)
            loss_train, __ = sess.run([cost, optimizer], 
                feed_dict=select_data('trn'))
            # Calculate batch accuracy
            loss_val = sess.run(cost, 
                feed_dict=select_data('val'))

            # best score check
            if loss_val <= loss_val_best:
                #Store scores
                loss_val_best = loss_val
                tolerance_count = 0
                self.best_i = i

                metric_list = [cost, precision, recall, fscore]
                self.best_trn_metrics = sess.run(
                    metric_list, feed_dict=select_data('trn'))
                self.best_val_metrics = sess.run(
                    metric_list, feed_dict=select_data('val'))
                self.best_tst_metrics = sess.run(
                    metric_list, feed_dict=select_data('tst'))
            else:
                # continuous condition
                if tolerance_count < FLAGS.n_tolerance and loss_val > 1e-5:
                    tolerance_count += 1
                elif self.best_i > 1000:
                    break

            # logs
            if i % FLAGS.display_step == 0:
                message = '  - iter {:4} --> trn, val = {:.3f}, {:.3f}'
                message = message.format(i, loss_train, loss_val)
                print(message, end='\r', flush=True)

        
        self.elapsed_time = time.time() - start_time
        message = '  - {} trained until {:4} --> f1 = {:.3f}, {:.3f}, {:.3f}'
        message = message.format(FLAGS.now, self.best_i, self.best_trn_metrics[-1], 
        self.best_val_metrics[-1], self.best_tst_metrics[-1])
        print(message)

        #print("\r")
        #print('  - best at iter {:4} with fsc {:03f}, {:03f}'.format(best_i, self.best_train_metrics[-1], self.best_test_metrics[-1]))
        # Close!!
        # sess.close()
        self.sess = sess


    def get_metrics(self):
        return self.best_trn_metrics, self.best_val_metrics, self.best_tst_metrics

    def clv(self, FLAGS):
        v = tf.trainable_variables()[0] #'conv1/weights'
        v_val = np.squeeze(v.eval(session=self.sess), axis=2)
        filter_list = np.dsplit(v_val, v_val.shape[2])
        return_str = ''
        for i, conv in enumerate(filter_list):
            conv_pos = np.squeeze(np.maximum(conv, 0), axis=2)
            w = np.mean(conv_pos, axis=0)
            d = np.absolute(w - np.median(w))
            clv = 100 * np.maximum(d - len(d) * np.median(d), 0)
            if np.max(clv) > 0:
                return_str += 'Filter # {:02}'.format(i)
                arg_clv = np.argsort(clv)[::-1]
                for arg in arg_clv:
                    if clv[arg] > 0:
                        return_str += '{}: {:.2f}'.format(FLAGS.sensor_list[arg], clv[arg])
                    else:
                        break
        return return_str

    def group_by_sensor_score(self, FLAGS, unified=True):
        v = tf.trainable_variables()[0] #'conv1/weights'
        v_val = np.squeeze(v.eval(session=self.sess), axis=2)
        filter_list = np.dsplit(v_val, v_val.shape[2])

        if unified is True:
            unified_filter = np.vstack(filter_list)
            unified_filter = np.squeeze(unified_filter, axis=2)
            sum_by_sensor = np.sqrt(np.sum(unified_filter**2, axis=0))
            arg_sum = np.argsort(sum_by_sensor)[::-1]
        else:
            filter_list = [np.squeeze(f) for f in filter_list]
            sum_by_sensor_list = [np.sqrt(np.sum(f**2, axis=0)) for f in filter_list]
            aggregated_by_filter_list = np.vstack(sum_by_sensor_list)
            sum_by_sensor = np.sum(aggregated_by_filter_list, axis=0)
            arg_sum = np.argsort(sum_by_sensor)[::-1]
        return_str = ''
        for arg in arg_sum:
            return_str += '{}({:.2f}),'.format(FLAGS.sensor_list[arg], sum_by_sensor[arg])
        for s in sum_by_sensor:
            return_str += '{},'.format(s)
        return return_str
            # for i, a_filter in enumerate(filter_list):
            #     a_filter = np.squeeze(a_filter, axis=2)
            #     sum_by_sensor = np.sqrt(np.sum(a_filter**2, axis=0))
            #     arg_sum = np.argsort(sum_by_sensor)[::-1]
            #     print('Filter # {:02}'.format(i))
            #     for arg in arg_sum:
            #         print('  - ', FLAGS.sensor_list[arg], ' : ', sum_by_sensor[arg])

    def save_conv_filter(self, FLAGS):
        model_name = '{}-{:02}-{}-{}'.format(
            FLAGS.recipe_no, FLAGS.step_no, FLAGS.device_id, FLAGS.model)

        save_dir_path = make_get_path('tf_weights', model_name, FLAGS.model_desc + FLAGS.now)

        v = tf.trainable_variables()[0] #'conv1/weights'
        v_name = v.name.replace(':','_').replace('/', '%')

        v_val = np.squeeze(v.eval(session=self.sess), axis=2)
        split = np.dsplit(v_val, v_val.shape[2])

        v_val = np.vstack(split)
        v_val_pos = np.maximum(v_val, 0)

        fname_pos = os.path.join(save_dir_path, v_name + '-pos.csv')
        np.savetxt(fname_pos, v_val_pos, delimiter=',')

        fname = os.path.join(save_dir_path, v_name + '.csv')
        np.savetxt(fname, v_val, delimiter=',')

    def save_results(self, FLAGS, param_dict):
        # record file existence check
        if not os.path.exists(FLAGS.record_fname):
            with open(FLAGS.record_fname, 'w') as fout:
                fout.write('recipe,step,device,model_name,')
                fout.write('group_c,n_conv,n_filter,window_size,')
                fout.write('pooling_size,pooling_stride,n_hidden_list,')
                fout.write('learning_rate,n_oversample,keeprate,minor_penalty,')
                fout.write('trn_loss,trn_prec,trn_rec,trn_f1,')
                fout.write('val_loss,val_prec,val_rec,val_f1,')
                fout.write('tst_loss,tst_prec,tst_rec,tst_f1,')
                fout.write('n_iter,elapsed_time,')
                fout.write('\n')
        # record performance
        with open(FLAGS.record_fname, 'a') as fout:
            fout.write('{},{},{},{},'.format(
                FLAGS.recipe_no, FLAGS.step_no, FLAGS.device_id, FLAGS.model))
            fout.write('{},{},{},{},'.format(
                param_dict.group_c, param_dict.n_conv, param_dict.n_filter, param_dict.window_size))
            fout.write('{},{},{},'.format(
                param_dict.pooling_size, param_dict.pooling_stride, str(param_dict.n_hidden_list).replace(',','/')))
            fout.write('{},{},{},{},'.format(
                FLAGS.learning_rate, FLAGS.n_oversample, FLAGS.keeprate, FLAGS.minor_weight))
            fout.write('{},{},{},{},'.format(*self.best_trn_metrics))
            fout.write('{},{},{},{},'.format(*self.best_val_metrics))
            fout.write('{},{},{},{},'.format(*self.best_tst_metrics))
            fout.write('{},{},'.format(self.best_i, self.elapsed_time))
            if FLAGS.model == 'group_cnn':
                fout.write(self.group_by_sensor_score(FLAGS, unified=False))
            elif FLAGS.model == 'group_cnn2':
                fout.write(self.group_by_sensor_score(FLAGS, unified=True))
            fout.write('\n')


    def save_weights(self, FLAGS):
        model_name = '{}-{:02}-{}-{}'.format(
            FLAGS.recipe_no, FLAGS.step_no, FLAGS.device_id, FLAGS.model)

        save_dir_path = make_get_path('tf_weights', model_name, FLAGS.model_desc + FLAGS.now)

        for v in tf.trainable_variables():
            if v.name[:5] == 'conv1':
                # print(v.name, ' ', v.get_shape())
                v_val = v.eval(session=self.sess)
                v_name = v.name.replace(':','_').replace('/', '%')
                
                # print(save_val.shape)
                if len(v_val.shape) > 2:
                    # print(v_val.shape)
                    v_val = np.squeeze(v_val, axis=2)
                    split = np.dsplit(v_val, v_val.shape[2])
                    # print(split[0].shape)
                    v_val = np.vstack(split)
                    # print(save_val.shape)
                    v_val_pos = np.maximum(v_val, 0)

                    fname_pos = os.path.join(save_dir_path, v_name + '-pos.csv')
                    np.savetxt(fname_pos, v_val_pos, delimiter=',')

                fname = os.path.join(save_dir_path, v_name + '.csv')
                np.savetxt(fname, v_val, delimiter=',')

        # with self.sess as sess:
        #     for v in tf.trainable_variables():
        #         # print(v.name, ' ', v.get_shape())
                
        #         if len(v.get_shape()) > 3 and v.get_shape()[3] > 1:
        #             for no in range(v.get_shape()[3]):
        #                 # print(no)
        #                 save_val = v.eval()[:, :, :, no]
                        
        #                 fname = v.name.replace(':','_').replace('/', '%')
        #                 fname = fname + str(no)
        #                 fname = '{}.csv'.format(fname)
        #                 fname = os.path.join(save_dir_path, fname)
        #                 np.savetxt(fname, save_val, delimiter=',')
        #         else:
        #             fname = v.name.replace(':','_').replace('/', '%')
        #             fname = '{}.csv'.format(fname)
        #             fname = os.path.join(save_dir_path, fname)
        #             np.savetxt(fname, v.eval(), delimiter=',')



    def close(self):
        self.sess.close()

