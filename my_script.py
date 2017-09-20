import os
import argparse
from multiprocessing import Process, Pool

num_iteration = 10

sh = (
  'python my_train.py {} {} {} '
  '--gpu_no={} --gpu_usage={} '
  '--model={} --wd={}'
)

model_list = [0, 1, 2, 3]

if __name__ == '__main__':
  # parse input parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('recipe_no', type=int, default=1,
                      help='recipe no')
  parser.add_argument('step_no', type=int, default=11,
                      help='step no')
  parser.add_argument('device_id', type=int, default=1,
                      help='device id')

  parser.add_argument('--model', type=int, default=0,
                      help='0/1/2/3')
  args, unparsed = parser.parse_known_args()

  # recipe_no, step_no, device_id = 1, 11, 1

  gpu_no = 0 if args.model % 2 == 0 else 1
  gpu_usage = 0.45

  if args.model == 0:
    wd = 0
    this_sh = sh.format(
        args.recipe_no, args.step_no, args.device_id, gpu_no, gpu_usage, args.model, wd)
    for _ in range(num_iteration):
      os.system(this_sh)

  else:
    wd_list = [0.01, 0.05, 0.1, 1]
    for wd in wd_list:
      this_sh = sh.format(
        args.recipe_no, args.step_no, args.device_id, gpu_no, gpu_usage, args.model, wd)
      for _ in range(num_iteration):
        os.system(this_sh)