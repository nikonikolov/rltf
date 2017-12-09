import gym

def get_monitor_wrapper(env):
  """
  Args:
    env: gym.Env. The wrapped environment.
  Returns:
    the `gym.wrappers.Monitor` around env

  Raises:
    `ValueError` if env is not wrapper by Monitor
  """

  currentenv = env
  while True:
    if "Monitor" in currentenv.__class__.__name__:
      return currentenv
    elif isinstance(env, gym.Wrapper):
      currentenv = currentenv.env
    else:
      raise ValueError("Couldn't find wrapper named Monitor")
