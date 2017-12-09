import argparse
import glob
import os
import random

import gym
import numpy      as np
import tensorflow as tf

import rltf.conf


def str2bool(v):
  """Parse command line bool argument"""
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def set_global_seeds(i):
  tf.set_random_seed(i)
  np.random.seed(i)
  random.seed(i)
  rltf.conf.SEED = i


def make_env(env_id, seed, model_dir, save_video, video_freq=None):
  """Create an instance of a gym environment, wrap it in a Monitor class and
  set seeds for the environment and for other modules (tf, np, random)

  Args:
    env_id: str. Full name of the gym environment
    seed: int. Seed for the environment and the modules
    model_dir: std. Path where videos from the Monitor class will be saved
    save_video: bool. If False, no videos will be recorded
    video_freq: int. Every `video_freq` episode will be recorded. If `None`,
      then every perfect cube episode number 1, 8, 27 ... 1000 will be recorded.
      After that, every 1000-th episode will be recorded
  Returns:
    The environment wrapped inside a Monitor class
  """

  # Set all seeds
  set_global_seeds(seed)

  env_file = model_dir + "Env.pkl"
  if os.path.isfile(env_file):
    return pickle_restore(env_file)

  gym_dir = model_dir + "gym_video"
  if save_video:
    if video_freq is None:
      video_callable = None
    else:
      video_callable = lambda e_id: e_id % video_freq == 0
  else:
    video_callable = False

  env = gym.make(env_id)
  env.seed(seed)
  env = gym.wrappers.Monitor(env, gym_dir, force=True, video_callable=video_callable)

  return env


def make_model_dir(model_type, env_id):
  """
  Args:
    model_type: python class or str. The class of the model
    env_id: str. The environment name
  Returns:
    The absolute path for the model directory
  """

  if isinstance(model_type, str):
    model_name  = model_type.lower()
  else:
    model_name  = model_type.__name__.lower()

  model_dir   = os.path.join(rltf.conf.MODELS_DIR,   model_name)
  model_dir   = os.path.join(model_dir,   env_id)

  # Get the number of existing models
  pattern     = model_dir + "_m*/"
  models      = glob.glob(pattern)

  # Get the number of the new model dir
  model_dir  += "_m" + str(len(models)+1)
  model_dir   = os.path.join(model_dir, "")

  # Create the directory for the model
  os.makedirs(model_dir)

  return model_dir
