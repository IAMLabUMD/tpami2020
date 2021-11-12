"""
utils
"""
import tensorflow as tf

import numpy as np
import subprocess as sp

ACCEPTABLE_AVAILABLE_MEMORY = 10000

# https://github.com/yselivonchyk/TensorFlow_DCIGN/blob/master/utils.py
def _output_to_list(output):
  return output.decode('ascii').split('\n')[:-1]


def get_idle_gpu(leave_unmasked=1, random=True):
  try:
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(command.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]

    if len(available_gpus) <= leave_unmasked:
      print('Found only %d usable GPUs in the system' % len(available_gpus))
      return -1

    if random:
      available_gpus = np.asarray(available_gpus)
      np.random.shuffle(available_gpus)

    gpu_to_use = available_gpus[0]
    print("Using GPU: ", gpu_to_use)
    
    return int(gpu_to_use)
    """
    # update CUDA variable
    gpus = available_gpus[:leave_unmasked]
    setting = ','.join(map(str, gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = setting
    print('Left next %d GPU(s) unmasked: [%s] (from %s available)'
          % (leave_unmasked, setting, str(available_gpus)))
    """
  except FileNotFoundError as e:
    print('"nvidia-smi" is probably not installed. GPUs are not masked')
    print(e)
    return -1
  except sp.CalledProcessError as e:
    print("Error on GPU masking:\n", e.output)
    return -1
