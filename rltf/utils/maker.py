import datetime
import logging
import os

import gym

from rltf.envs        import MaxEpisodeLen
from rltf.monitoring  import Monitor
from rltf.utils       import rltf_conf
from rltf.utils       import rltf_log
from rltf.utils       import seeding


logger = logging.getLogger(__name__)


def _make_env(env_id, seed, model_dir, wrap, mode, log_period, video_period, max_ep_steps):
  if "Roboschool" in env_id:
    import roboschool

  if "Bullet" in env_id:
    import pybullet_envs

  env = gym.make(env_id)
  if seed >= 0:
    env.seed(seed)

  # NOTE: Episode steps limit wrapper must be set before any other wrapper, including the Monitor.
  # Otherwise, statistics of reported reward will be wrong
  if max_ep_steps is not None:
  # if max_ep_steps is not None and max_ep_steps > 0:
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=max_ep_steps)
    env = MaxEpisodeLen(env, max_episode_steps=max_ep_steps)

  if wrap is not None:
    env = wrap(env, mode=mode)

  # Apply monitor at the very top
  env = Monitor(env, model_dir, log_period, mode, video_period)

  return env


def _set_seeds(seed):
  seeding.set_random_seed(seed)  # Set RLTF seed
  seeding.set_global_seeds()     # Set other module's seeds


def make_envs(env_id, seed, model_dir, log_period_train, log_period_eval,
              video_period=None, wrap=None, max_ep_steps_train=None, max_ep_steps_eval=None):
  """Create two instances of a gym environment, wrap them in a Monitor class and
  set seeds for the environments and for other modules (tf, np, random). Both
  environments are not allowed to be in dual mode. One env is for train, one for eval
  Args:
    env_id: str. Full name of the gym environment
    seed: int. Seed for the environment and the modules
    model_dir: std. Path where videos from the Monitor class will be saved
    video_period: int. Every `video_period` episode will be recorded. If `None`,
      then the monitor default is used. If `<=0`, then no videos are recorded
    wrap: function. Must take as arguments the environment and its mode and wrap it.
    max_ep_steps_train: int. Set a bound on the max steps in a training episode. If None, no limit
    max_ep_steps_eval: int. Set a bound on the max steps in an evaluation episode. If None, no limit
  Returns:
    Tuple of the environments, each wrapped inside a Monitor class
  """

  _set_seeds(seed)

  env_train = _make_env(env_id, seed,   model_dir, wrap, 't', log_period_train, video_period, max_ep_steps_train)
  env_eval  = _make_env(env_id, seed+1, model_dir, wrap, 'e', log_period_eval,  video_period, max_ep_steps_eval)

  return env_train, env_eval


def make_model_dir(args, base=rltf_conf.MODELS_DIR):
  """Construct the correct absolute path of the model and create the directory.
  Args:
    args: argparse.ArgumentParser. The command-line arguments
    base: str. The absolute path of the directory where all models are saved
  Returns:
    The absolute path for the model directory
  """

  # Get the model, the env, values of restore and reuse
  model_type  = args.model
  env_id      = args.env_id
  restore_dir = args.restore_model
  reuse_dir   = args.load_model

  # If restoring, do not create a new directory
  if restore_dir is not None:
    model_dir = restore_dir

  # If evaluating, create a subdirectory
  elif args.mode == 'eval':
    assert reuse_dir is not None
    model_dir = os.path.join(reuse_dir, "eval/")
    os.makedirs(model_dir)

  # Create a new model directory
  else:
    model_id    = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    model_id    = env_id + "_" + model_id
    model_name  = model_type.lower()

    model_dir   = os.path.join(base,      model_name)
    model_dir   = os.path.join(model_dir, model_id)
    model_dir   = os.path.join(model_dir, "")

    # Create the directory for the model
    os.makedirs(model_dir)

  # Configure the logger
  rltf_log.conf_logs(model_dir, args.log_lvl, args.log_lvl)

  return model_dir
