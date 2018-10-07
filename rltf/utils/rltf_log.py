import datetime
import logging.config
import os
import subprocess

import rltf.conf

param_logger = logging.getLogger(rltf.conf.PARAM_LOGGER_NAME)


def conf_logs(model_dir, stdout_lvl="DEBUG", file_lvl="DEBUG"):

  run_file    = os.path.join(model_dir, "run.log")
  param_file  = os.path.join(model_dir, "params.txt")

  conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters':
      {
      'default':
        {
          'format': '[%(levelname)s] %(name)s: %(message)s'
        },
      'info_formatter':
        {
          'format': '%(message)s'
        },
      },
    'handlers':
      {
        # Console debug file handler
        'console': {
          'level': stdout_lvl,
          'class': 'logging.StreamHandler',
          'formatter': 'default',
          'stream': 'ext://sys.stdout'
        },
        # Runtime file handler
        'run_file': {
          'level': file_lvl,
          'class': 'logging.FileHandler',
          'formatter': 'default',
          'filename': run_file,
        },
        # Parameter file handler
        'param_file': {
          'level': 'INFO',
          'class': 'logging.FileHandler',
          'formatter': 'info_formatter',
          'filename': param_file,
        },
        # Normal stdout info and stats
        'std_info': {
          'level': 'INFO',
          'class': 'logging.StreamHandler',
          'formatter': 'info_formatter',
          'stream': 'ext://sys.stdout'
        },
      },
    'loggers':
      {
      # All loggers
      '':
        {
          'handlers': ['console', 'run_file'],
          'level': 'DEBUG',
          'propagate': True
        },
      # Parameter file logger
      rltf.conf.PARAM_LOGGER_NAME:
        {
          'handlers': ['std_info', 'run_file', 'param_file'],
          'level': 'INFO',
          'propagate': False
        },
      # Trianing stat reports logger
      rltf.conf.STATS_LOGGER_NAME:
        {
          'handlers': ['std_info', 'run_file'],
          'level': 'INFO',
          'propagate': False
        },
      }
    }


  logging.config.dictConfig(conf)

  # Log the git diff
  try:
    diff = subprocess.check_output(["git", "diff"], cwd=rltf.conf.PROJECT_DIR)
    diff = diff.decode("utf-8")
    if diff != "":
      with open(os.path.join(model_dir, "git.diff"), 'w') as f:
        f.write(diff)
  except subprocess.CalledProcessError:
    # git repo not initialized
    pass


def log_params(params, args):
  """Log the runtime parameters for the model to a file on disk
  Args:
    params: list. Each entry must be a tuple of (name, value). Value can also
      be any time of object, but it should have an implementation of __str__
    args: ArgumentParser. The command line arguments
  """
  params  = pad_log_data(params, sort=True)
  date    = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  commit  = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=rltf.conf.PROJECT_DIR)
  commit  = commit.decode("utf-8").strip("\n")
  branch  = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=rltf.conf.PROJECT_DIR)
  branch  = branch.decode("utf-8").strip("\n")

  param_logger.info("")
  param_logger.info("TIME: %s", date)
  param_logger.info("GIT COMMIT: %s", commit)
  param_logger.info("GIT BRANCH: %s", branch)
  param_logger.info("")
  param_logger.info("AGENT CONFIG:")
  for k, v in params:
    param_logger.info(k + ": " + str(v))
  param_logger.info("")

  if args is not None:
    param_logger.info("COMMAND LINE ARGS:")
    args = args.__dict__.items()
    args = format_tabular(args, 30)
    for s, v in args:
      param_logger.info(s.format(v))
    param_logger.info("")


def format_tabular(data, value_width=15, sort=True):
  """
  Args:
    data: list of tuples. The first member of the tuple must be str, the last
      must be either a lambda or the variable to be printed
    value_width: int. The max number of chars that the value of the last member
      of the tuple can take
  Returns:
    list of tuples. Calling `print(s.format(v)) for s,v in data`, will result in
      tabular print. If `v` is lambda, it must be evaluated before printing
  """
  data = pad_log_data(data, sort)
  width = len(data[0][0]) + 6 + value_width

  # Check if the values are computed with a lambda or variable evaluation
  if callable(data[0][-1]):
    border = ("-" * width + "{}", lambda *args, **kwargs: "")
  else:
    border = ("-" * width + "{}", "")

  if    len(data[0]) == 2:
    data = [("| " + s + "| {:<" + str(value_width) + "} |", v) for s, v in data]
  elif  len(data[0]) == 3:
    data = [("| " + s + "| {:<" + str(value_width) + f + "} |", v) for s, f, v in data]
  else:
    raise ValueError("Tuple must have len 2 or 3")

  data = [(s, str(v)) if v is None else (s, v) for s, v in data]
  data = [border] + data + [border]

  return data


def pad_log_data(data, sort):
  """
  Args:
    data: list of tuples. The first member of the tuple must be str, the
      rest can be anything.
  Returns:
    The list, with the strs padded with space chars in order to align in tabular way
  """
  if sort:
    data  = sorted(data, key=lambda tup: tup[0])
  sizes = [len(t[0]) for t in data]
  pad   = max(sizes) + 2
  data  = [(t[0].ljust(pad), *t[1:]) for t in data]
  return data
