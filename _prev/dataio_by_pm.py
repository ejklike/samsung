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

import logging
logging.basicConfig(
    filename='./log/dataio_by_pm.log',
    filemode='w',
    level=logging.INFO)

from itertools import product
import pickle

import pandas as pd
import numpy as np

class RecipeData(object):
    def __init__(self, recipe_no):
        # import sensor name
        sensor_fname = './data/RECIPE{}_sensor_id.csv'.format(recipe_no)
        sensor_list = pd.read_csv(sensor_fname, header=None, delimiter=',')[0].tolist()

        # import meta data
        meta_fname = './data/RECIPE{}_material_id.csv'.format(recipe_no)
        meta_header = ['material_id', 'device_id', 'datetime', 'fault']
        meta_df = pd.read_csv(meta_fname, header=None, delimiter=',', names=meta_header)
        meta_df.set_index('material_id', inplace=True)    

        self.pm = dict(
            PM1={},
            PM2={},
            PM6={}
        )

        for step_no in range(1, 25 + 1):
            print('RECIPE {} - STEP {}'.format(recipe_no, step_no))
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

            # # filter sensors containing only one values for each wafer ()
            # def is_std_vaild_by_wafer(df, sensor_index, threshold):
            #     sensor_flow = df.iloc[:, sensor_index::sensor_size]
            #     summary_by_wafer = sensor_flow.apply(pd.DataFrame.describe, axis=1)
            #     return summary_by_wafer['std'].max() >= threshold
            # threshold = 0.5
            # sensor_mask = [is_std_vaild_by_wafer(df, s, threshold) for s in range(sensor_size)]
            # df = df.iloc[:, sensor_mask * time_size]
            # sensor_size = df.shape[1] // time_size

            # split by device id
            def filter_data_by_device_id(df, meta_df, device_id):
                filtered_meta_df = meta_df[meta_df.device_id == device_id]
                filtered_wafer_list = filtered_meta_df.index.tolist()
                filtered_x = df.filter(items=filtered_wafer_list, axis=0)
                filtered_y = meta_df.fault.filter(items=filtered_wafer_list, axis=0).values
                return dict(
                    X=filtered_x,
                    y=filtered_y,
                    wafer_list=filtered_wafer_list,
                    sensor_size=sensor_size,
                    time_size=time_size,
                )

            for device_id in ['PM1', 'PM2', 'PM6']:
                self.pm[device_id][step_no] = filter_data_by_device_id(df, meta_df, device_id)

if __name__ == '__main__':
    n_recipe = 2

    rcp_data = {}
    for i in range(1, n_recipe + 1):
        rcp_data[i] = RecipeData(i)

    pickle.dump(rcp_data, open('./data_by_pm_all_sensor.p', 'wb'))