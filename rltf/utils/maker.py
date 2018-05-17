import datetime
import logging
import os

import gym

import rltf.conf
import rltf.utils.seeding
import rltf.monitoring


logger = logging.getLogger(__name__)


def _make_env(env_id, seed, model_dir, video_callable, mode, dual_mode, max_ep_steps):
  if ("Roboschool") in env_id:
    import roboschool

  monitor_dir = os.path.join(model_dir, "env_monitor")

  env = gym.make(env_id)
  if seed >= 0:
    env.seed(seed)

  # NOTE: Episode steps limit wrapper must be set before the Monitor. Otherwise, statistics of
  # reported reward will be wrong
  if max_ep_steps is not None:
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_ep_steps)
  env = rltf.monitoring.Monitor(env, monitor_dir, video_callable, mode, dual_mode)

  return env


def _get_video_callable(video_freq):
  if video_freq is None:
    video_callable = None
  elif video_freq > 0:
    video_callable = lambda e_id: e_id % video_freq == 0
  else:
    video_callable = False
  return video_callable


def _set_seeds(seed):
  rltf.utils.seeding.set_random_seed(seed)  # Set RLTF seed
  rltf.utils.seeding.set_global_seeds()     # Set other module's seeds


def make_env(env_id, seed, model_dir, video_freq=None, wrap=None, max_ep_steps=None):
  """Create an instance of a gym environment, wrap it in a Monitor class and
  set seeds for the environment and for other modules (tf, np, random). The
  monitor of the environment is set in dual mode
  Args:
    env_id: str. Full name of the gym environment
    seed: int. Seed for the environment and the modules
    model_dir: std. Path where videos from the Monitor class will be saved
    video_freq: int. Every `video_freq` episode will be recorded. If `None`,
      then the monitor default is used. If `<=0`, then no videos are recorded
    max_ep_steps: int. Set a bound on the max steps in an episode. If None, no limit
  Returns:
    The environment wrapped inside a Monitor class
  """

  _set_seeds(seed)
  video_callable = _get_video_callable(video_freq)
  env = _make_env(env_id, seed, model_dir, video_callable, 't', True, max_ep_steps)
  env = wrap(env)
  return env


def make_envs(env_id, seed, model_dir, video_freq=None, wrap=None, max_ep_steps=None):
  """Create two instances of a gym environment, wrap them in a Monitor class and
  set seeds for the environments and for other modules (tf, np, random). Both
  environments are not allowed to be in dual mode. One env is for train, one for eval
  Args:
    env_id: str. Full name of the gym environment
    seed: int. Seed for the environment and the modules
    model_dir: std. Path where videos from the Monitor class will be saved
    video_freq: int. Every `video_freq` episode will be recorded. If `None`,
      then the monitor default is used. If `<=0`, then no videos are recorded
    max_ep_steps: int. Set a bound on the max steps in an episode. If None, no limit
    wrap: function. Must take as argument the environment and wrap it.
  Returns:
    Tuple of the environments, each wrapped inside a Monitor class
  """

  _set_seeds(seed)
  video_callable = _get_video_callable(video_freq)

  env_train = _make_env(env_id, seed,   model_dir, video_callable, 't', False, max_ep_steps)
  env_eval  = _make_env(env_id, seed+1, model_dir, video_callable, 'e', False, max_ep_steps)

  env_train = wrap(env_train)
  env_eval  = wrap(env_eval)

  return env_train, env_eval


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
