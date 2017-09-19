import argparse
from datetime import datetime
import time

# do not print warning logs
# https://stackoverflow.com/questions/35911252
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import sys
from itertools import product

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf

import models
import custom_models
from dataio_by_pm import RecipeData
from wrapper_cv import Model

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def binary_to_dummies(arr):
    arr = arr.reshape(-1, 1)
    return np.concatenate((1-arr, arr), axis=1)

# normalize and reshape
def normalize_and_reshape(X, time_size, sensor_size):
    """in: (wafer, time * sensor)
        out: (wafer, time, sensor)
    """
    hsplit = np.hsplit(X, time_size) # (wafer, sensor) * time
    merge = np.vstack(hsplit) # (wafer * time, sensor)
    
    scaled = StandardScaler().fit_transform(merge)
    # scaled = MinMaxScaler().fit_transform(merge)

    vsplit = np.vsplit(scaled, time_size) # (wafer, sensor) * time
    scaled_data = np.hstack(vsplit) # (wafer, )
    reshaped_data = scaled_data.reshape(-1, time_size, sensor_size)
    return reshaped_data


def main(__):
    # load data
    with open(FLAGS.file_name, 'rb') as data:
        rcp_data = pickle.load(data)[FLAGS.recipe_no]
        step_data = rcp_data.pm[FLAGS.device_id][FLAGS.step_no]

    # data size
    FLAGS.time_size = step_data['time_size']
    FLAGS.sensor_size = step_data['sensor_size']
    FLAGS.sensor_list = [v[-4:] for v in step_data['X'].columns[:FLAGS.sensor_size]]

    # data
    X = step_data['X'].values
    y = step_data['y']

    # raw data (np array)
    X = normalize_and_reshape(
        X, FLAGS.time_size, FLAGS.sensor_size)
    
    # split data: generate cv index
    if FLAGS.n_splits > 1:
        kfold = StratifiedKFold(
            n_splits=FLAGS.n_splits, random_state=FLAGS.random_state)
        kfold_index_list = list(kfold.split(X, y))
        FLAGS.val_size = 1/(FLAGS.n_splits)
    else:
        tmp_X = step_data['X']
        tmp_X = tmp_X.reset_index(drop=True)
        _X_trn, _X_tst, __, __ = train_test_split(tmp_X, step_data['y'],
                        test_size=FLAGS.test_size, 
                        random_state=FLAGS.random_state,
                        stratify=step_data['y'])
        kfold_index_list = [(_X_trn.index.values, _X_tst.index.values)]
        FLAGS.val_size = FLAGS.test_size
    y = binary_to_dummies(y)

    step_info_msg = '*** recipe{}-step{:2}-device{}: ' + str(X.shape)
    step_info_msg = step_info_msg.format(FLAGS.recipe_no, FLAGS.step_no, FLAGS.device_id)
    print(step_info_msg)
    print(FLAGS.sensor_list)
        
    best_param_list = []
    best_score_list = []
    elapsed_time_list = []
    epoch_list = []

    for n_iter in range(FLAGS.num_repeat):
        print('=============== repeat {}/{} ================='.format(n_iter + 1, FLAGS.num_repeat))

        start_time = time.time()

        # param graid
        best_score, best_param, best_clv_str = 0, {}, ''
        for model_no, param_dict in enumerate(ParameterGrid(FLAGS.param_grid)):
            msg = 'model# {} {}'.format(model_no, param_dict)
            print(msg)

            FLAGS.model_desc = ''
            for k, v in param_dict.items():
                FLAGS.model_desc += '{}_{}&'.format(k, v)
            # do cv and get avg test score
            avg_tst_score = 0
            for cv_no, (trn_idx, tst_idx) in enumerate(kfold_index_list):
                FLAGS.now = datetime.now().strftime("%Y%m%d %H%M%S")

                # split data by cv index
                X_cv_trn, X_cv_tst = X[trn_idx], X[tst_idx]
                y_cv_trn, y_cv_tst = y[trn_idx], y[tst_idx]

                # oversample fault in training
                def oversample(X, y, n_oversample):
                    fault_idx = np.where(y[:, 1]==1)[0]
                    X_fault = X[fault_idx, :]
                    y_fault = y[fault_idx, :]
                    for _ in range(n_oversample):
                        X = np.concatenate((X, X_fault), axis=0)
                        y = np.concatenate((y, y_fault), axis=0)
                    return X, y
                X_cv_trn, y_cv_trn = oversample(X_cv_trn, y_cv_trn, FLAGS.n_oversample)

                #split trn to get val data
                X_trn, X_val, y_trn, y_val = train_test_split(
                    X_cv_trn, y_cv_trn, 
                    test_size=FLAGS.val_size, 
                    random_state=FLAGS.random_state, 
                    stratify=y_cv_trn)
                FLAGS.batch_size = X_trn.shape[0]

                # training model
                model = Model(dotdict(param_dict), FLAGS)
                model.train(X_trn, y_trn, X_val, y_val, X_cv_tst, y_cv_tst, FLAGS)
                trn_metrics, val_metrics, tst_metrics = model.get_metrics()
                # add test metric
                if not np.isnan(tst_metrics[-1]):
                    avg_tst_score += tst_metrics[-1] / len(kfold_index_list)
                # model.save_conv_filter(FLAGS)
                model.save_weights(FLAGS)
                if FLAGS.model == 'group_cnn':
                    clv_str = model.group_by_sensor_score(FLAGS, unified=False)
                elif FLAGS.model == 'group_cnn2':
                    clv_str = model.group_by_sensor_score(FLAGS, unified=True)
                else:
                    clv_str = model.clv(FLAGS)
                epoch_list.append(model.best_i)

                model.save_results(FLAGS, dotdict(param_dict))



                del model
            
            # message = '  ->> average{:.3f}'
            # check best_model
            if avg_tst_score > best_score:
                best_score, best_param, best_clv_str = avg_tst_score, param_dict, clv_str

        elapsed_time = time.time() - start_time
        
        msg = '*** best score and its configuration: {:.3f} @ {}, elapsed time: {:.1f}'
        msg = msg.format(best_score, str(best_param), elapsed_time)
        print(step_info_msg)
        print(msg)
        print(best_clv_str)

        best_score_list.append(best_score)
        best_param_list.append(best_param)
        elapsed_time_list.append(elapsed_time)

    zipped_list = zip(range(FLAGS.num_repeat), best_score_list, best_param_list, elapsed_time_list)
    for i, score, param, elapsed_time in zipped_list:
        print('#{:02}: {:3f} @ {}, time={:.1f}'.format(i, score, param, elapsed_time))

    mean = np.mean(best_score_list)
    std = np.std(best_score_list)
    print('--> f1: avg {:.3f}, std {:.3f}'.format(mean, std))

    mean = np.mean(elapsed_time_list)
    std = np.std(elapsed_time_list)
    print('--> time: avg {:.1f}, std {:.1f}'.format(mean, std))

    mean = np.mean(epoch_list)
    std = np.std(epoch_list)
    print('--> epoch: avg {:.1f}, std {:.1f}'.format(mean, std))




if __name__ == '__main__':
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rnn',
                        help='rnn/cnn/lstm')

    parser.add_argument('recipe_no', type=int, default=1,
                        help='recipe no')
    parser.add_argument('step_no', type=int, default=1,
                        help='step no')
    parser.add_argument('device_id', type=str, default='PM6',
                        help='device id')
    FLAGS, unparsed = parser.parse_known_args()

    # Assume that you have 12GB of GPU memory and want to allocate ~4GB: 0.333
    FLAGS.gpu_usage = 0.90

    # file
    # FLAGS.file_name = './data_by_pm_var0.5.p'
    FLAGS.file_name = './data_by_pm_all_sensor.p'

    # record
    FLAGS.record_fname = './record.csv'

    # learning parameters
    FLAGS.max_iter = 30000
    FLAGS.learning_rate = 1e-4
    FLAGS.n_tolerance = 2000
    FLAGS.keeprate = 0.95
    # FLAGS.batch_size = 50
    FLAGS.n_oversample = 10
    FLAGS.num_repeat = 5

    # display option
    FLAGS.display_step = 100

    # data split for cv
    FLAGS.n_splits = 1
    FLAGS.test_size = 2/10
    FLAGS.random_state = 42

    # grid search params
    # param_grid_rnn = dict(
    #         n_hidden=[50, 100, 200, 300],
    #         n_rnn_unit=[50, 100, 200, 300]
    # )

    # if FLAGS.device_id == 'PM1':
    #     param_grid_cnn = dict(
    #         n_hidden_list = [[50]],#, [50, 50]],#, [30, 30], [50, 50], [50, 30]],
    #         window_size=[5],
    #         n_conv=[1],#, 2],
    #         n_filter=[10],#, 20], #[5, 10, 20],
    #         pooling_size=[2],
    #         pooling_stride=[1]
    #     )
    # elif FLAGS.device_id == 'PM2':
    #     param_grid_cnn = dict(
    #         n_hidden_list = [[50, 50]],#, [50, 50]],#, [30, 30], [50, 50], [50, 30]],
    #         window_size=[5],
    #         n_conv=[1],#, 2],
    #         n_filter=[10],#, 20], #[5, 10, 20],
    #         pooling_size=[2],
    #         pooling_stride=[1]
    #     )
    # elif FLAGS.device_id == 'PM6':
    #     param_grid_cnn = dict(
    #         n_hidden_list = [[50], [50, 50]],#, [50, 50]],#, [30, 30], [50, 50], [50, 30]],
    #         window_size=[5],
    #         n_conv=[1],#, 2],
    #         n_filter=[10],#, 20], #[5, 10, 20],
    #         pooling_size=[2],
    #         pooling_stride=[1]
    #     )
    param_grid_cnn = dict(
        # n_hidden=[30],#, 100],
        n_hidden_list = [[], [10], [30], [50], [100], [30,50],[50, 50],[50,30],[30,30]],
        window_size=[3,5,7],
        n_conv=[1, 2],
        n_filter=[5, 10, 20],
        pooling_size=[2],
        pooling_stride=[1, 2]
    )
    if FLAGS.model == 'rnn':
        FLAGS.model_name = models.rnn
        FLAGS.param_grid = param_grid_rnn

    # elif FLAGS.model == 'lstm':
    #     FLAGS.model_name = models.lstm
    #     FLAGS.param_grid = param_grid_rnn

    # elif FLAGS.model == 'cnn':
    #     FLAGS.model_name = models.cnn
    #     FLAGS.param_grid = param_grid_cnn

    elif FLAGS.model == 'group_cnn':
        FLAGS.minor_weight = 1
        FLAGS.model_name = custom_models.group_cnn
        FLAGS.param_grid = param_grid_cnn
        FLAGS.param_grid['group_c'] = [0.01, 0.05, 0.1]# #0.01, 0.1]#, 0.02, 0.05] #0.001,  #, 0.5, 1
 
    elif FLAGS.model == 'group_cnn2':
        FLAGS.minor_weight = 1
        FLAGS.model_name = custom_models.group_cnn2
        FLAGS.param_grid = param_grid_cnn
        FLAGS.param_grid['group_c'] = [0.01, 0.05, 0.1]#, 0.1]

    # elif FLAGS.model == 'group_cnn3':
    #     FLAGS.minor_weight = 1
    #     FLAGS.model_name = custom_models.group_cnn3
    #     FLAGS.param_grid = param_grid_cnn
    #     FLAGS.param_grid['group_c'] = [0.01, 0.05, 0.1]

    elif FLAGS.model == 'fdc_cnn':
        FLAGS.minor_weight = 1
        FLAGS.model_name = custom_models.fdc_cnn
        FLAGS.param_grid = param_grid_cnn
        FLAGS.param_grid['group_c'] = [0]
    
    # elif FLAGS.model == 'group_cnn_mw':
    #     FLAGS.model_name = custom_models.group_cnn_multi_windows
    #     FLAGS.param_grid = param_grid_cnn
    #     FLAGS.param_grid['group_c'] = [0.01, 0.05, 0.1] #0.001,  #, 0.5, 1
    
    else:
        raise "No matching result with your input model name."

    tf.app.run(main=main)#, argv=[sys.argv[1]] + unparsed)