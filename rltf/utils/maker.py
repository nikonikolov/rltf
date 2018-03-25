import datetime
import logging
import os

import gym

import rltf.conf
import rltf.utils.seeding
import rltf.monitoring
# from    rltf.utils import set_global_seeds


logger = logging.getLogger(__name__)


def make_env(env_id, seed, model_dir, save_video, video_freq=None):
  """Create an instance of a gym environment, wrap it in a Monitor class and
  set seeds for the environment and for other modules (tf, np, random)

  Args:
    env_id: str. Full name of the gym environment
    seed: int. Seed for the environment and the modules
    model_dir: std. Path where videos from the Monitor class will be saved
    save_video: bool. If False, no videos will be recorded
    video_freq: int. Every `video_freq` episode will be recorded. If `None`,
      then the monitor default is used
  Returns:
    The environment wrapped inside a Monitor class
  """

  if ("Roboschool") in env_id:
    import roboschool

  # Set all seeds
  rltf.utils.seeding.set_global_seeds(seed)

  monitor_dir = os.path.join(model_dir, "env_monitor")
  if save_video:
    if video_freq is None:
      video_callable = None
    else:
      video_callable = lambda e_id: e_id % video_freq == 0
  else:
    video_callable = False

  env = gym.make(env_id)
  env.seed(seed)
  # env = gym.wrappers.Monitor(env, monitor_dir, video_callable=video_callable)
  env = rltf.monitoring.Monitor(env, monitor_dir, video_callable=video_callable)

  return env


def make_model_dir(model_type, env_id, dest=rltf.conf.MODELS_DIR):
  """
  Args:
    model_type: python class or str. The class of the model
    env_id: str. The environment name
    dest: str. The absolute path of the directory where all models are saved
  Returns:
    The absolute path for the model directory
  """

  if isinstance(model_type, str):
    model_name  = model_type.lower()
  else:
    model_name  = model_type.__name__.lower()

  model_dir   = os.path.join(dest,      model_name)
  model_dir   = os.path.join(model_dir, env_id)

  model_id    = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
  model_dir   = model_dir + "_" + model_id
  model_dir   = os.path.join(model_dir, "")

  # Create the directory for the model
  logger.info('Creating model directory %s', model_dir)
  os.makedirs(model_dir)

  return model_dir
