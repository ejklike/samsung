"""
recipe_data
  - pm_data 1
    - step_data 1
    - step_data 2
    - ...
  - pm_data 2
    - step_data 1
    - step_data 2
    - ...
  - ...

"""

from itertools import product
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_data(recipe_no, step_no, devicde_id):
    # import sensor name
    sensor_fname = './data/RECIPE{}_sensor_id.csv'.format(recipe_no)
    sensor_list = pd.read_csv(sensor_fname, header=None, delimiter=',')[0].tolist()

    # import meta data
    meta_fname = './data/RECIPE{}_material_id.csv'.format(recipe_no)
    meta_header = ['material_id', 'device_id', 'datetime', 'fault']
    meta_df = pd.read_csv(meta_fname, header=None, delimiter=',', names=meta_header)
    meta_df.set_index('material_id', inplace=True)    

    # import step data
    fname = './data/RECIPE{}_STEP{:02}.csv'.format(recipe_no, step_no)
    material_list = meta_df.index
    df = pd.read_csv(fname, header=None, delimiter=',', names=material_list).transpose()

    # set column names
    sensor_size = len(sensor_list)
    time_size = df.shape[1] // sensor_size
    time_sensor_prod = product(range(1, time_size + 1), sensor_list)
    column_names = ['t{:03}_{}'.format(t, s) for t, s in time_sensor_prod]
    df.columns = column_names

    # filter timestamps containing null sensor values
    is_null_column = np.sum(df.isnull().values, axis=0) > 0
    if is_null_column.any():
        for t in range(time_size):
            start, end = sensor_size * t, sensor_size * (t + 1)
            col_idx = list(range(start, end))
            if df.iloc[:, col_idx].isnull().values.any():
                break
        df = df.iloc[:, :start]
        time_size = df.shape[1] // sensor_size

    # split by device id
    def filter_data_by_device_id(df, meta_df, device_id):
        filtered_meta_df = meta_df[meta_df.device_id == device_id]
        filtered_wafer_list = filtered_meta_df.index.tolist()
        filtered_x = df.filter(items=filtered_wafer_list, axis=0).as_matrix()
        filtered_y = meta_df.fault.filter(items=filtered_wafer_list, axis=0).as_matrix()
        print(filtered_wafer_list[0])
        fault_idx = np.where(filtered_y==1)[0]
        print([filtered_wafer_list[idx] for idx in fault_idx])
        return filtered_x, filtered_y

    signals, labels = filter_data_by_device_id(df, meta_df, device_id)
    signals, labels = signals.astype(np.float32), labels.astype(int)

    time_size = 15
    sensor_size = 65

    def normalize_and_reshape(X, time_size, sensor_size):
        """in: (wafer, time * sensor)
            out: (wafer, time, sensor)
        """
        # print(X[0, :])
        hsplit = np.hsplit(X, time_size) # (wafer, sensor) * time
        # print(hsplit[0][0, :])
        merge = np.vstack(hsplit) # (time * wafer, sensor)
        
        scaled = StandardScaler().fit_transform(merge)
        # scaled = MinMaxScaler().fit_transform(merge)
        # scaled = merge

        vsplit = np.vsplit(scaled, time_size) # (wafer, sensor) * time
        scaled_data = np.hstack(vsplit) # (wafer, time * sensor)
        reshaped_data = scaled_data.reshape(-1, time_size, sensor_size)# (wafer, time, sensor)
        
        # print(reshaped_data[0, 14, :])
        return reshaped_data

    signals = normalize_and_reshape(signals, time_size, sensor_size)

    def binary_to_dummies(arr):
        arr = arr.reshape(-1, 1)
        return np.concatenate((1-arr, arr), axis=1)

    labels = binary_to_dummies(labels)
    print(np.sum(labels, axis=0))

    return dict(signals=signals, labels=labels)


if __name__ == '__main__':
    recipe_no = 2
    step_no = 11
    device_id = 'PM6'
    
    args = (recipe_no, step_no, device_id)
    fname = './{}-{}-{}.p'.format(*args)
    print(fname, end='... ')
    
    data = get_data(*args)
    pickle.dump(data, open(fname, 'wb'))
    print('saved.')