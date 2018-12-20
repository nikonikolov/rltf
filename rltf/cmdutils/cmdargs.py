import argparse
import os

from rltf.cmdutils          import ArgSpec
from rltf.cmdutils          import LambdaArgSpec
from rltf.cmdutils.defaults import * #pylint: disable=wildcard-import,unused-wildcard-import


def str2bool(v):
  """Parse command line bool argument"""
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(model_choices):
  """Parse both known and unknown command line arguments. Automatically fetches the default arguments
  for the selected model and overrides the defaults with data from the unknown arguments.
  Args:
    model_choices: list of strings with the allowed model choices
  Returns:
    tuple of (agent_kwargs, args)
    agent_kwargs: dict with arguments to be passed directly to the agent
    args: namespace with the parsed known command line arguments
  """
  s2b = str2bool

  # Create the parser and add args
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--env-id',        required=True,   type=str,   help='full environment name')
  parser.add_argument('--model',         required=True,   type=str,   choices=model_choices)

  parser.add_argument('--seed',          default=42,      type=int,   help='global seed; random if <=0')
  parser.add_argument('--mode',          default='train', type=str,   choices=['train', 'play'])
  parser.add_argument('--n-plays',       default=0,       type=int,   help='number of runs in play mode')
  parser.add_argument('--log-lvl',       default='INFO',  type=str,   help='logger lvl')
  parser.add_argument('--plot-video',    default=False,   type=s2b,   help='add model plots to videos')

  # Optional arguments
  parser.add_argument('--restore-model', default=None,    type=str,
    help='(optional) directory path of existing model to restore and continue training')
  parser.add_argument('--load-model',    default=None,    type=str,
    help='(optional) directory path of existing model whose weights will be loaded initially')
  parser.add_argument('--load-regex',    default=None,    type=str,
    help='(optional) regex for matching the set of vars to load with --load-model')

  # Parse only the known args
  args, extra_args = parser.parse_known_args()

  # Verify the correctness of the known args
  args = verify_args(args)

  # Fetch the default arguments for the model
  model_kwargs = get_args(args.model)

  # Update the defaults with the extra command line arguments
  model_kwargs = parse_extra_args(extra_args, model_kwargs)

  # Build the defaults
  model_kwargs = build_kwargs(model_kwargs)

  # Get the required command line arguments to be passed to the agent
  cmd_kwargs = dict(
    n_plays=args.n_plays,
    plot_video=args.plot_video,
    load_model=args.load_model,
    load_regex=args.load_regex,
  )

  # Initialize the agent kwargs: merge the overriden defaults and the applicable command-line arguments
  agent_kwargs = {**model_kwargs, **cmd_kwargs}

  return agent_kwargs, args


def parse_extra_args(extra_args, kwargs):
  """Parse the extra command line arguments and update any overriden defaults
  Args:
    extra_args: list of command line arguments
    kwargs: dict of default arguments
  Returns:
    The updated kwargs
  """
  for arg in extra_args:
    assert arg.startswith('--')
    assert '=' in arg, "Cannot parse arg {}".format(arg)

    key, value = arg[2:].split('=', 1)

    # Check if argument is for nested ArgSpec assignment
    if '.' in key:
      key, *subkeys = key.split(".")
    else:
      subkeys = None

    assert key in kwargs, "Unknown model argument {}".format(key)

    # Update a base type argument or a complete ArgSpec definition
    if subkeys is None:
      kwargs[key] = eval(value)

    # Update the subkey of an ArgSpec
    else:
      builder = kwargs[key]

      # Get the correct ArgSpec builder, even if nested
      if isinstance(builder, ArgSpec):
        builder.override(subkeys, value)
      # Update the lambda so that arguments are truly overriden
      elif builder.__name__ == "<lambda>":
        kwargs[key] = LambdaArgSpec(builder, subkeys, value)
      else:
        raise TypeError("Only arguments of types 'lambda' or 'ArgSpec' can be overriden in a nested way.")

  return kwargs


def build_kwargs(kwargs):
  for key, value in kwargs.items():
    if isinstance(value, ArgSpec):
      kwargs[key] = value()
    elif callable(value) and not isinstance(value, LambdaArgSpec) and value.__name__ == "<lambda>":
      kwargs[key] = LambdaArgSpec(value)
  return kwargs


def verify_args(args):
  # Only one of args.restore_model and args.load_model can be set
  assert not (args.restore_model is not None and args.load_model is not None)

  # When in play mode, model needs to be loaded
  if args.mode == 'play':
    assert args.load_model is not None
    assert args.n_plays > 0

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

  return args
