import datetime
import os

import gym

from rltf.envs        import MaxEpisodeLen
from rltf.utils       import rltf_conf
from rltf.utils       import rltf_log
from rltf.utils       import seeding


def get_env_maker(env_id, seed, wrap=None, max_ep_steps_train=None, max_ep_steps_eval=None, **wrap_kwargs):
  """Create an environment maker function
  Args:
    env_id: str or callable. If str, full name of a registered gym, roboschool or pybullet
      env. If callable, must return a new env instance.
    seed: int. Seed for the environment and the modules
    wrap: function. Must take as arguments the environment and its mode and wrap it.
    max_ep_steps_train: int. A limit on the max steps in a training episode.
    max_ep_steps_eval: int. A limit on the max steps in an evaluation episode.
    wrap_kwargs: dict. Keyword arguments that will be passed to the wrapper
  Returns:
    callable which takes the mode of an env and builds a new enviornment instance
  """

  # Set the global seed. Note that once the seed it set, multiple calls to this do not afect randomness
  seeding.set_random_seeds(seed)

  # Create a variable which tracks the seeds passed to environments.
  # This is to prevent environments from having the same seed, which will cause unwanted correlation.
  env_seed = int(seed)

  if isinstance(env_id, str):
    if "Roboschool" in env_id:
      import roboschool

    if "Bullet" in env_id:
      import pybullet_envs

    make = lambda: gym.make(env_id)

  elif callable(env_id):
    make = env_id

  else:
    raise ValueError("You must provide a str or a function for 'env_id', "
                     "not {}: {}".format(type(env_id), env_id))


  def make_env(mode):
    nonlocal env_seed

    # Make the environment
    env = make()

    if env_seed >= 0:
      # Increment seed to avoid producing identical environments
      env_seed += 1
      env.seed(env_seed)

    # NOTE: Wrapper for episode steps limit must be set before any other wrapper
    if mode == 't' and max_ep_steps_train is not None:
      env = MaxEpisodeLen(env, max_episode_steps=max_ep_steps_train)
    elif mode == 'e' and max_ep_steps_eval is not None:
      env = MaxEpisodeLen(env, max_episode_steps=max_ep_steps_eval)

    if wrap is not None:
      env = wrap(env, mode, **wrap_kwargs)

    return env

  return make_env


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
