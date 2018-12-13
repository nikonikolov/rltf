import datetime
import logging.config
import os
import subprocess

from rltf.utils import rltf_conf


param_logger = logging.getLogger(rltf_conf.PARAM_LOGGER_NAME)
stats_logger = logging.getLogger(rltf_conf.STATS_LOGGER_NAME)

COLORS = dict(
  gray=37,
  red=31,
  green=32,
  yellow=93,
  blue=94,
  magenta=35,
  cyan=36,
  white=97,
)


def conf_logs(model_dir, stdout_lvl="DEBUG", file_lvl="DEBUG"):

  run_file    = os.path.join(model_dir, "run.log")

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
        # Runtime file handler
        'info_run_file': {
          'level': file_lvl,
          'class': 'logging.FileHandler',
          'formatter': 'info_formatter',
          'filename': run_file,
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
      rltf_conf.PARAM_LOGGER_NAME:
        {
          'handlers': ['std_info', 'info_run_file'],
          'level': 'INFO',
          'propagate': False
        },
      # Trianing stat reports logger
      rltf_conf.STATS_LOGGER_NAME:
        {
          'handlers': ['std_info', 'info_run_file'],
          'level': 'INFO',
          'propagate': False
        },
      }
    }


  logging.config.dictConfig(conf)

  # Log the git diff
  try:
    diff = subprocess.check_output(["git", "diff"], cwd=rltf_conf.PROJECT_DIR)
    diff = diff.decode("utf-8")
    if diff != "":
      with open(os.path.join(model_dir, "git.diff"), 'w') as f:
        f.write(diff)
  except subprocess.CalledProcessError:
    # git repo not initialized
    pass


def colorize(string, color, bold=False, highlight=False):
  attr = []
  code = COLORS[color]
  if highlight: code += 10
  attr.append(str(code))
  if bold: attr.append('1')
  return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def log_params(params, args):
  """Log the runtime parameters for the model to a file on disk
  Args:
    params: list. Each entry must be a tuple of (name, value). Value can also
      be any time of object, but it should have an implementation of __str__
    args: ArgumentParser. The command line arguments
  """

  # Log date and time and git commit and branch
  date    = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  commit  = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=rltf_conf.PROJECT_DIR)
  commit  = commit.decode("utf-8").strip("\n")
  branch  = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=rltf_conf.PROJECT_DIR)
  branch  = branch.decode("utf-8").strip("\n")

  param_logger.info("")
  param_logger.info("TIME: %s", date)
  param_logger.info("GIT COMMIT: %s", commit)
  param_logger.info("GIT BRANCH: %s", branch)
  param_logger.info("")

  # Build the list of data that will be logged
  data = {**vars(args), **dict(params)}
  data = data.items()

  # Log the parameters
  param_logger.info("PARAMETERS:")
  data = format_tabular(data, 80)
  for s, v in data:
    try:
      param_logger.info(s.format(v))
    except TypeError:
      param_logger.info(s.format(str(v)))
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
  data = _pad_keys_tabular(data, sort)
  width = len(data[0][0]) + 6 + value_width

  # Values available directly
  if len(data[0]) == 2:
    hborder = ("-" * width + "{}", "")
    data    = [("| " + s + "| {:<" + str(value_width) + "} |", v) for s, v in data]
  # Values available from calling a lambda
  elif len(data[0]) == 3:
    hborder = ("-" * width + "{}", lambda *args, **kwargs: "")
    data    = [("| " + s + "| {:<" + str(value_width) + f + "} |", v) for s, f, v in data]
  else:
    raise ValueError("Tuple must have len 2 or 3")


  data = [(s, str(v)) if v is None else (s, v) for s, v in data]
  data = [hborder] + data + [hborder]

  return data


_DUMP_TABULAR = []


def log_tabular(name, value):
  _DUMP_TABULAR.append((name, value))


def dump_tabular(logger=stats_logger):
  global _DUMP_TABULAR
  data = _DUMP_TABULAR
  # Format in tabular way
  data = format_tabular(data, sort=False)
  # Dump the data
  logger.info("")
  for s, v in data:
    logger.info(s.format(v))
  logger.info("")
  _DUMP_TABULAR = []


def _pad_keys_tabular(data, sort):
  """Pad only the key fields in data (i.e. the strs) in a tabular way, such that they
  all take the same amount of characters
  Args:
    data: list of tuples. The first member of the tuple must be str, the rest can be anything.
  Returns:
    list with the strs padded with space chars in order to align in tabular way
  """
  if sort:
    data  = sorted(data, key=lambda tup: tup[0])
  sizes = [len(t[0]) for t in data]
  pad   = max(sizes) + 2
  data  = [(t[0].ljust(pad), *t[1:]) for t in data]
  return data
