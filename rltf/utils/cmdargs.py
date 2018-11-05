import argparse
import logging
import os


logger = logging.getLogger(__name__)


def str2bool(v):
  """Parse command line bool argument"""
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(custom_args):
  """Parse command line arguments. Apart from the custom arguments received as input, adds some
    default arguments which are needed for a general program. Performs additional checks for the
    validity of the arguments.
  Args:
    custom_args: list of tuples `(str, dict)`. Each tuple is the name of the argument and a dictionary
      with keyword arguments for `argparse.ArgumentParser.add_argument()`
  Returns:
    The result of `argparse.ArgumentParser.parse_args()`
  """

  common_args = [
    ('--gamma',        dict(default=0.99,   type=float,   help='discount factor')),
    ('--seed',         dict(default=42,     type=int,     help='seed for the run; not set if <=0')),
    ('--mode',         dict(default='train',type=str,     choices=['train', 'eval'])),
    ('--n-evals',      dict(default=0,      type=int,     help='# of separate runs in evaluation mode')),
    ('--log-lvl',      dict(default='INFO', type=str,     help='logger lvl')),
    ('--plot-video',   dict(default=False,  type=str2bool,help='if True, add model plots to video')),
    ('--log-period',   dict(default=10000,  type=int,     help='how often to log stats in # *agent* steps')),
    ('--save-period',  dict(default=0,      type=int,     help='how often to save the model in # \
      *agent* steps; if <=0, model is saved only at the end')),
    ('--save-buf',     dict(default=True,   type=str2bool,help='whether to save the buffer')),
    ('--video-period', dict(default=500,    type=int,     help='how often to record videos in # \
      *episodes*; if <=0, do not record any video')),
    ('--restore-model',dict(default=None,   type=str,     help='path to existing dir; if set, will \
      continue training with the network and the env from this dir')),
    ('--load-model',   dict(default=None,   type=str,     help='path to existing dir; if set, will \
      only load the network weights from the dir, but write a new model and train it on a new env')),
    ('--extra-info',   dict(default="",     type=str,     help='extra info, not captured by command \
      line args that should be added to the program log')),
    ('--confirm-kill', dict(default=False,  type=str2bool,help='if True, Ctrl+C has to be confirmed')),
    ('--load-regex',   dict(default=None,   type=str,     help='regex for matching vars to load; \
      used with --load-model')),
  ]

  common_args_names = { arg[0]: i for i, arg in enumerate(common_args)}
  custom_args_names = { arg[0]: i for i, arg in enumerate(custom_args)}

  # Remove overriden duplicates
  for arg in custom_args_names:
    if arg in common_args_names:
      logger.warning("Removing the default command line option %s and replacing it with %s",
                     common_args[common_args_names[arg]], custom_args[custom_args_names[arg]])
      common_args.pop(common_args_names[arg])

  # Merge all arguments
  arg_spec = custom_args + common_args

  # Create the parser and add args
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  for arg in arg_spec:
    parser.add_argument(arg[0], **arg[1])

  # Parse arguments and perform checks
  args = parser.parse_args()

  # log_period must be positive
  assert args.log_period > 0

  assert args.gamma > 0 and args.gamma <= 1

  # Only one of args.restore_model and args.load_model can be set
  assert not (args.restore_model is not None and args.load_model is not None)

  # When in eval mode, model needs to be loaded
  if args.mode == 'eval':
    assert args.load_model is not None
    assert args.n_evals > 0

  if args.restore_model is not None:
    args.restore_model = os.path.abspath(args.restore_model)
    assert os.path.exists(args.restore_model)
    assert os.path.basename(args.restore_model).startswith(args.env_id)

  elif args.load_model is not None:
    args.load_model = os.path.abspath(args.load_model)
    assert os.path.exists(args.load_model)
    assert os.path.exists(os.path.join(args.load_model, "tf"))

  if args.load_regex is not None:
    assert args.load_model is not None

  # # Grad clip and huber loss cannot be simultaneously set
  # try:
  #   if args.grad_clip is not None:
  #     assert args.grad_clip > 0
  #     assert not args.huber_loss
  # except AttributeError:
  #   pass

  return args
