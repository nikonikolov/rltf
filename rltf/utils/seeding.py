import random
import numpy      as np
import tensorflow as tf

import rltf.conf

def set_global_seeds(i):
  tf.set_random_seed(i)
  np.random.seed(i)
  random.seed(i)
  rltf.conf.SEED = i
