import random
import numpy      as np
import tensorflow as tf

def set_global_seeds(seed):
  if seed < 0:
    return

  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
