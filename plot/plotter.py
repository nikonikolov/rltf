import argparse
import os
import pprint
import warnings

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import dataio

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR     = os.path.join(PROJECT_DIR, "figures")


def parse_cmd_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Configuration arguments
  parser.add_argument('--conf',       required=True,    type=str,   help='path for conf file')
  parser.add_argument('--filename',   default=None,     type=str,   help='output filename; default is `conf`')

  # Matplotlib arguments
  parser.add_argument('--ncols',      default=2,        type=int,   help='number of cols in the fig')
  parser.add_argument('--nrows',      default=2,        type=int,   help='number of rows in the fig')
  parser.add_argument('--width',      default=1920,     type=int,   help='width of a subplot')
  parser.add_argument('--height',     default=1080,     type=int,   help='height of a subplot')
  parser.add_argument('--subscale',   default=0.5,      type=float, help='scale factor for subplot size')
  # parser.add_argument('--fwidth',     default=None,     type=int,   help='width of the figure')
  # parser.add_argument('--fheight',    default=None,     type=int,   help='height of the figure')
  parser.add_argument('--fontsize',   default=16,       type=int,   help='fontsize for all plots')
  parser.add_argument('--shrink',     default=0.9,      type=float, help='shrink factor to fit legend')
  parser.add_argument('--pad',        default=3,        type=int,   help='pad between subplots and fig margins')
  parser.add_argument('--hpad',       default=3,        type=int,   help='horizonal pad between subplots')
  parser.add_argument('--wpad',       default=3,        type=int,   help='vertical pad between subplots')

  # Model runtime arguments
  parser.add_argument('--max-step',   default=30*10**6, type=int,   help='max train step for data')
  parser.add_argument('--eval-freq',  default=250000,   type=int,   help='freq of eval in # *agent* steps')
  parser.add_argument('--eval-len',   default=125000,   type=int,   help='len of eval in # *agent* steps')
  parser.add_argument('--n-eps',      default=100,      type=int,   help='number of eps to average')

  parser.add_argument('--tablemax',   default=True,     type=bool,  help='bold max scores in table output')

  args = parser.parse_args()

  return args

def parse_args():
  args = parse_cmd_args()
  conf = args.conf + ".json"
  conf = dataio.read_conf(conf)

  # Default to the conf name for output files
  if args.filename is None:
    args.filename = args.conf

  # TODO:
  # - If custom conf file is provided, need to make sure path is correct
  # - If the conf file contains custom arguments, they must be overwritten in args
  if "args" in conf and len(conf["args"].keys()) > 0:
    newargs  = conf["args"]
    argnames = vars(args)
    for newarg, value in newargs.items():
      if newarg in argnames:
        oldval = getattr(args, newarg)
        warnings.warn("Overwriting command line argument {}={} with conf file value {}".format(
                      newarg, oldval, value))
      else:
        warnings.warn("Ignoring conf file argument {}={}".format(newarg, value))

      assert type(value) == type(oldval)
      setattr(args, newarg, value)

  # Compute correct values for Figure
  args.width  = int(args.width  * args.subscale)
  args.height = int(args.height * args.subscale)

  # Compute max step in terms of eval and a step scaling factor
  # NOTE: Need to keep max step in terms of eval due to TensorBoard
  assert args.eval_freq % args.eval_len == 0
  args.step_scale = int(args.eval_freq / args.eval_len)
  args.max_step   = int(args.max_step / args.step_scale)

  args.conf = conf

  return args


def make_figure(args):
  dpi = 100.0

  # TODO: If custom figure values, take them into account
  ncols   = args.ncols
  nrows   = args.nrows
  width   = args.width  * ncols
  height  = args.height * nrows

  fig  = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
  axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)
  # Make a list of the axes - otherwise it is 2D array
  axes = [ax for r in axes for ax in r]

  # Apply the style properties to every subplot
  for ax in axes:
    ax.grid()
    ax.tick_params()
    # ax.ticklabel_format(style='sci', axis='both', scilimits=(0,10000))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

  fig.tight_layout(pad=args.pad, h_pad=args.hpad, w_pad=args.wpad)

  return fig, axes


def add_legend(fig, axes, envs, shrink):
  # Get line handles and labels and remove duplicates
  # handles, labels = fig.gca().get_legend_handles_labels()
  handles = [ax.get_legend_handles_labels()[0] for ax in axes]
  labels  = [ax.get_legend_handles_labels()[1] for ax in axes]
  handles = [handle for hlist in handles for handle in hlist]
  labels  = [label  for llist in labels  for label  in llist]
  by_label = OrderedDict(zip(labels, handles))

  # Place in lower-right corner when no plot there
  if len(axes) > len(envs):
    # Get the center position of the next subplot
    bbox = axes[len(envs)].get_position().get_points()
    x, y = (bbox[0][0]+bbox[1][0])/2.0, (bbox[0][1]+bbox[1][1])/2.0
    # Put the legend in the center of the next subplot
    fig.legend(by_label.values(), by_label.keys(), loc="center", bbox_to_anchor=(x, y))

    # Use different legend label size and make lines bolder
    # leg = fig.legend(by_label.values(), by_label.keys(), loc="center", bbox_to_anchor=(x, y),
    #                  prop={'size': 32})
    # for handle in leg.legendHandles:
    #   handle.set_linewidth(10.0)

  # Otherwise place as a row on top and shrink subplots
  else:
    fig.legend(by_label.values(), by_label.keys(), loc="upper left", ncol=len(by_label.keys()))
    fig.subplots_adjust(top=shrink)


def limit_model_steps(model_dir, model_data, args):
  """Assumes model_data is in proper format:
    - len(scores_rews) == len(scores_steps) == len(scores_inds)
    - each index in the score arrays corresponds to a single eval run
    - the values in score_inds correspond to correct indices in ep_rews and ep_lens
  Args:
    model_dir: str. The model directory
    model_data: dict. The model data
    args: ArgumentParser. The command-line arguments
  Returns:
    dict. Contains the input data limited to args.max_step. Steps are scaled to match train agent steps
  """
  scores_steps  = model_data["scores_steps"]
  max_step      = args.max_step
  step_scale    = args.step_scale

  # TODO: This assumes that max_step is contained in scores_step. Otherwise, additional index is kept
  inds = np.where(scores_steps >= max_step)[0]
  try:
    i = inds[0]+1
    model_data["scores_steps"] = model_data["scores_steps"][:i] * step_scale
    model_data["scores_rews"]  = model_data["scores_rews"][:i]
    model_data["scores_inds"]  = model_data["scores_inds"][:i]

    # Remember that indices are exclusive
    i = model_data["scores_inds"][-1]

    model_data["ep_lens"] = model_data["ep_lens"][:i]
    model_data["ep_rews"] = model_data["ep_rews"][:i]

  # Append garbage data if max_step not reached
  except IndexError:
    warnings.warn("The recorded data for model '%s' has not reached max_step=%d. Appending -inf values"
                  % (model_dir, max_step))

    ep_lens       = list(model_data["ep_lens"])
    ep_rews       = list(model_data["ep_rews"])
    scores_steps  = list(model_data["scores_steps"] * step_scale)
    scores_rews   = list(model_data["scores_rews"])
    scores_inds   = list(model_data["scores_inds"])

    period = args.eval_len

    while True:
      s = scores_steps[-1] + period
      if s > max_step:
        break
      ep_rews.append(-float("inf"))
      ep_lens.append(0)
      scores_steps.append(s)
      scores_rews.append(-float("inf"))
      scores_inds.append(len(ep_rews))

    model_data["ep_lens"]      = np.asarray(ep_lens, dtype=np.int32)
    model_data["ep_rews"]      = np.asarray(ep_rews, dtype=np.float32)
    model_data["scores_steps"] = np.asarray(scores_steps, dtype=np.int32)
    model_data["scores_rews"]  = np.asarray(scores_rews, dtype=np.float32)
    model_data["scores_inds"]  = np.asarray(scores_inds, dtype=np.int32)

  return model_data


def n_episodes_averages(model_data, n_eps=100):
  """Average episode rewards in a window of n_eps"""

  ep_rews = model_data["ep_rews"]
  inds    = model_data["scores_inds"]
  # rews    = [np.mean(ep_rews[i-n_eps:i]) for i in inds]
  rews    = [ep_rews[i-n_eps:i] for i in inds]
  rews    = [np.mean(eps) if len(eps) > 0 else -float("inf") for eps in rews]
  rews    = np.asarray(rews)

  assert len(rews)  > 0

  return dict(scores_rews=model_data["scores_rews"],
              scores_steps=model_data["scores_steps"],
              mean_rews=rews,
             )

def filter_tb_data(x, y, args, model_dir):
  assert len(x) == len(y)
  steps, rews = [], []

  max_step    = args.max_step
  step_scale  = args.step_scale
  period      = args.eval_len

  for s, r, s1, r1 in zip(x, y, x[1:], y[1:]):
    if s > max_step:
      break
    if s % period == 0:
      # Avoid duplicates
      if len(steps) == 0 or (len(steps) > 0 and steps[-1] != s):
        steps.append(s)
        rews.append(r)
    if s % period == period-5000:
      s = s + 5000
      r = (r+r1)/2
      # Avoid duplicates
      if len(steps) == 0 or (len(steps) > 0 and steps[-1] != s):
        steps.append(s)
        rews.append(r)

  # Check for correct parsing
  if len(rews) == 0 or len(steps) == 0:
    print(len(rews), len(steps))
    print(x)
    print(y)
    raise AssertionError

  # Append garbage y to max_step until max_step
  if steps[-1] < max_step:
    warnings.warn("The recorded data for model '%s' has not reached max_step=%d. Appending -inf values"
                  % (model_dir, max_step))
    while True:
      s = (len(steps)+1)*period
      if s > max_step:
        break
      steps.append(s)
      rews.append(-float("inf"))

  # Double check correct step values
  for i, s in enumerate(steps):
    if (i+1)*period != s:
      print((i+1)*period, s)
      print(steps)
      print(x)
      print((rews[i]+rews[i-1])/2)
      raise AssertionError

  steps = np.asarray(steps) * step_scale
  rews  = np.asarray(rews)
  return dict(scores_steps=steps, mean_rews=rews)


def process_run(model_dir, args):
  """Read the data of a single model and prepare it for plotting. Cuts any data above args.max_steps.
  If the model does not have scores saved, the data is recovered from TensorBoard files.
  Args:
    model_dir: str. Name of the model in format model-name/env-name_run-date
    args:
  Returns:
    `dict` containing the data, ready to be plotted
  """
  print("Processing '%s'" % model_dir)

  model_path  = dataio.get_model_dir(model_dir, args)
  model_data  = dataio.read_model_data(model_path)
  rews        = model_data["scores_rews"]
  steps       = model_data["scores_steps"]
  inds        = model_data["scores_inds"]
  use_tb      = rews is None or steps is None or inds is None

  # Compute model data from stats
  if use_tb:
    assert args.n_eps == 100
    x, y       = dataio.read_tb_file(model_path, tag="eval/mean_ep_rew")
    model_data = filter_tb_data(x, y, args, model_dir)
  # Compute model data from TensorBoard
  else:
    model_data = limit_model_steps(model_dir, model_data, args)
    model_data = n_episodes_averages(model_data, n_eps=args.n_eps)

  return model_data


def process_group(groups, args):
  """Given grouped models, read and process the data for each one
  Args:
    groups: dict. Structure as returned from group_models
    fn: lambda. Takes model_dir as arguments and returns a dictionary of the processed model data
  Returns:
    dict. Has the same structure as groups, except that model directory names are substituted with 
      data read from the directory
  """
  for env, labels in groups.items():
    for label, runs in labels.items():
      # Read and process the data for every model in the list
      # models_data = {model_dir: fn(model_dir) for model_dir in runs}
      model_runs = [process_run(model_dir, args) for model_dir in runs]
      groups[env][label] = model_runs
  return groups


def group_models(legend):
  """Given a legend with models grouped by label, groups the models based on environments and labels
  Args:
    legend: dict. The value of the "legend" key in the configuration file
  Returns:
    `dict` with grouped models. Each key is an environment name and each value is a `dict`. Each key in
    the latter `dict` is a label for the matplotlib legend. The corresponding value is a list of model
    directories containing run data for the label and the environment. The idea is that if the size of the
    list is more than 1, then the entries correspond to runs with different seeds of the same model
    configuration in the same env
  """
  groups = dict()
  for label, data in legend.items():
    model_dirs = data["models"]
    for mdir in model_dirs:
      # Get the environment
      env = dataio.get_env_name(mdir)
      if env not in groups:
        # Use OrderedDict so that the order of labels when plotted is preserved - helps for overlay issues
        groups[env] = OrderedDict()
        # groups[env] = dict()
      # Add the label for this group in the environment models
      if label not in groups[env]:
        # groups[env][label] = {"color":, "data":}
        groups[env][label] = []
      # Append the model directory as part of this environment and label
      groups[env][label].append(mdir)
  return groups


def plot_model(axes, x, y, label, color):
  """
  Args:
    axes: matplotlib.axes.Axes. Axes object which should handle plotting
    x: np.array. x-coordinates for every entry of `y`
    y: list of `np.array`s. If len==1, plot a single line. Otherwise plot confidence interval and average
    label: str. Label for the plotted line
    color: str. MPL color for the plotted line
  """
  axes.plot(x, y["mean"], label=label, color=color)
  if "lo" in y and "hi" in y:
    axes.fill_between(x, y["lo"], y["hi"], color=color, alpha=0.5)


def average_models(x, y):
  if len(y) == 1:
    max_score = np.amax(y[0])
    mean = y[0]
    assert len(x) == len(mean)
    return max_score, dict(mean=mean)
  else:
    lo = np.amin(y, axis=0)
    hi = np.amax(y, axis=0)
    mean = np.mean(y, axis=0)
    assert len(x) == len(hi)
    assert len(x) == len(lo)
    assert len(x) == len(mean)
    max_score = np.amax(mean)
    return max_score, dict(mean=mean, lo=lo, hi=hi)


def plot_figure(args):
  """Read data, process it, create a Figure and plot the data
  Args:
    args: ArgumentParser. Contains the command-line arguments. Must also have member `conf` which holds
    the configuration
  Returns:
    tuple (Figure, dict). First entry is the plotted figure, second entry is a dict of the best scores
  """

  conf    = args.conf
  legend  = conf["legend"]

  groups  = group_models(legend)
  print("Groups:")
  pprint.pprint(groups)
  print()
  groups  = process_group(groups, args)
  envs    = sorted(groups.keys())
  scores  = dict()

  # TODO: Add functionality for plotting on separate figures
  fig, axes = make_figure(args)

  # Make unnecessary subplots not visible
  if len(axes) > len(envs):
    for i in range(len(envs),len(axes)):
      axes[i].set_visible(False)

  # Plot each figure
  for env, ax in zip(envs, axes):
    scores[env] = dict()
    # Set the title to the environment name
    ax.set_title(label=env)
    labels = groups[env]
    # Traverse legends so that labels are always plotted in the same order
    # for model in legend:
      # runs = labels[model]
    for label, runs in labels.items():
      color = legend[label]["color"]
      x = runs[0]["scores_steps"]
      y = [mdata["mean_rews"] for mdata in runs]
      print("Processing env: '%s', label: '%s'" % (env, label))
      score, y = average_models(x, y)
      plot_model(ax, x, y, label, color)
      scores[env][label] = score

  add_legend(fig, axes, envs, args.shrink)

  return fig, scores


def main():
  # Process args and conf
  args = parse_args()

  # Set the global fontsize for all plots
  plt.rc('font', size=args.fontsize)

  # Plot figure
  fig, scores = plot_figure(args)

  # Save figure
  pngfile = os.path.join(FIG_DIR, args.filename + ".png")
  plt.savefig(pngfile)

  # Save scores
  txtfile = os.path.join(FIG_DIR, args.filename)
  dataio.save_scores(scores, txtfile, args)

  plt.show()


if (__name__ == "__main__"):
  main()
