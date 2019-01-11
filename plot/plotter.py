import argparse
import os
import pprint
import warnings

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import dataio
import dataproc

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR     = os.path.join(PROJECT_DIR, "figures")

def _warning(message, category=UserWarning, filename='', lineno=-1, file=None, line=None):
  print("[WARNING]: %s" % message)

warnings.showwarning = _warning


def parse_cmd_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Configuration arguments
  parser.add_argument('--conf',       required=True,    type=str,   help='path for conf file')
  parser.add_argument('--filename',   default=None,     type=str,   help='output filename; default is `conf`')
  parser.add_argument('--score-mode', default=None,     type=str,   help='what statistic to plot',
                      choices=["mean_score", "n_eps"])
  parser.add_argument('--run',        default=None,     type=str,   help='plot or log table',
                      choices=["plot", "log"])

  # Matplotlib arguments
  parser.add_argument('--ncols',      default=2,        type=int,   help='number of cols in the fig')
  parser.add_argument('--nrows',      default=2,        type=int,   help='number of rows in the fig')
  parser.add_argument('--width',      default=1920,     type=int,   help='width of a subplot')
  parser.add_argument('--height',     default=1080,     type=int,   help='height of a subplot')
  parser.add_argument('--subscale',   default=0.5,      type=float, help='scale factor for subplot size')
  # parser.add_argument('--fwidth',     default=None,     type=int,   help='width of the figure')
  # parser.add_argument('--fheight',    default=None,     type=int,   help='height of the figure')
  parser.add_argument('--fontsize',   default=20,       type=int,   help='fontsize for all plots')
  parser.add_argument('--linewidth',  default=2.0,      type=float, help='linewidth for all plots')
  parser.add_argument('--shrink',     default=0.9,      type=float, help='shrink factor to fit legend')
  parser.add_argument('--pad',        default=3,        type=int,   help='pad between subplots and fig margins')
  parser.add_argument('--wpad',       default=3.0,      type=float, help='horizonal pad between subplots')
  parser.add_argument('--hpad',       default=3.0,      type=float, help='vertical pad between subplots')

  # Model runtime arguments
  parser.add_argument('--max-step',   default=50*10**6, type=int,   help='max train step for data')
  parser.add_argument('--tb-tag',     default=None,     type=str,   help='TensorBoard tag to read')
  parser.add_argument('--np-data',    default=None,     type=str,   help='Read train or eval npy data', choices=["t", "e"])

  parser.add_argument('--period',     default=None,     type=int,   help='filter data with this period')
  parser.add_argument('--boldmax',    default=True,     type=bool,  help='bold max scores in table output')
  parser.add_argument('--tablestd',   default=False,    type=bool,  help='Add std of max scores in table output')

  args = parser.parse_args()

  return args


def parse_args():
  args = parse_cmd_args()

  assert args.conf.endswith(".yml") or args.conf.endswith(".yaml")
  conf = dataio.read_conf(args.conf)

  # Default to the conf name for output files
  if args.filename is None:
    args.filename = args.conf

  # Read arguments from conf file - overwrite cmd values
  if "args" in conf and len(conf["args"].keys()) > 0:
    newargs  = conf["args"]
    argnames = vars(args)
    for newarg, value in newargs.items():
      print(newarg, value)
      if newarg in argnames:
        oldval = getattr(args, newarg)
        warnings.warn("Overwriting command line argument {}={} with conf file value {}".format(
                      newarg, oldval, value))
      else:
        warnings.warn("Ignoring conf file argument {}={}".format(newarg, value))

      if oldval is not None:
        assert type(value) == type(oldval)
      setattr(args, newarg, value)

  # Compute correct values for Figure
  args.width  = int(args.width  * args.subscale)
  args.height = int(args.height * args.subscale)

  # Assert that the mode for reading data is provided
  assert (args.tb_tag is None) != (args.np_data is None)

  args.conf = conf

  return args


def make_figure(args, nplots):
  dpi = 100.0

  ncols   = args.ncols
  nrows   = args.nrows
  width   = args.width  * ncols
  height  = args.height * nrows

  fig  = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
  axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)
  # Make a list of the axes - otherwise it is 2D array
  axes = [ax for r in axes for ax in r]

  # Make unnecessary subplots not visible
  if len(axes) < nplots:
    raise ValueError("Number of available subplots is less than the amount of data to be plotted")
  elif len(axes) > nplots:
    for i in range(nplots,len(axes)):
      axes[i].set_visible(False)

  # Apply the style properties to every subplot
  for ax in axes:
    ax.grid()
    ax.tick_params()
    # ax.ticklabel_format(style='sci', axis='both', scilimits=(0,10000))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

  fig.tight_layout(pad=args.pad, h_pad=args.hpad, w_pad=args.wpad)

  return fig, axes


def add_legend(fig, axes, nplots, shrink, linewidth):
  # Get line handles and labels and remove duplicates
  # handles, labels = fig.gca().get_legend_handles_labels()
  handles = [ax.get_legend_handles_labels()[0] for ax in axes]
  labels  = [ax.get_legend_handles_labels()[1] for ax in axes]
  handles = [handle for hlist in handles for handle in hlist]
  labels  = [label  for llist in labels  for label  in llist]
  by_label = OrderedDict(zip(labels, handles))

  # Place in lower-right corner when no plot there
  if len(axes) > nplots:
    # Get the center position of the next subplot
    bbox = axes[nplots].get_position().get_points()
    x, y = (bbox[0][0]+bbox[1][0])/2.0, (bbox[0][1]+bbox[1][1])/2.0
    # Put the legend in the center of the next subplot
    leg = fig.legend(by_label.values(), by_label.keys(), loc="center", bbox_to_anchor=(x, y))

    # Use different legend label size and make lines bolder
    # leg = fig.legend(by_label.values(), by_label.keys(), loc="center", bbox_to_anchor=(x, y),
    #                  prop={'size': 32})

  # Otherwise place as a row on top and shrink subplots
  else:
    leg = fig.legend(by_label.values(), by_label.keys(), loc="upper left", ncol=len(by_label.keys()))
    fig.subplots_adjust(top=shrink)

  # # For testing
  # leg = fig.legend(by_label.values(), by_label.keys(), loc="upper left", ncol=len(by_label.keys()))
  # fig.subplots_adjust(top=shrink)

  for handle in leg.legendHandles:
    handle.set_linewidth(linewidth*2)


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

  path = dataio.get_model_dir(model_dir, args)
  datawrap = dataproc.DataWrapper(path, args.max_step, tb_tag=args.tb_tag, data_type=args.np_data, log_period=args.period)
  datawrap.read_data()
  datawrap.data.compute_y(mode=args.score_mode)
  data = datawrap.get_data()

  return data


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


def plot_model(axes, x, y, label, color, linewidth):
  """
  Args:
    axes: matplotlib.axes.Axes. Axes object which should handle plotting
    x: np.array. x-coordinates for every entry of `y`
    y: list of `np.array`s. If len==1, plot a single line. Otherwise plot confidence interval and average
    label: str. Label for the plotted line
    color: str. MPL color for the plotted line
  """
  axes.plot(x, y["mean"], label=label, color=color, linewidth=linewidth)
  if y["lo"] is not None and y["hi"] is not None:
    axes.fill_between(x, y["lo"], y["hi"], color=color, alpha=0.3)


def plot_histogram(axes, x, y, label, color, linewidth):
  # https://github.com/tensorflow/tensorboard/blob/1.5/tensorboard/plugins/distribution/vz_distribution_chart/vz-distribution-chart.ts
  # opacities = [0.0912, 0.5436, 0.5992, 0.766, 0.766, 0.5992, 0.5436, 0.0912]
  opacities = [0.0912, 0.5436, 0.5992, 0.766, 0.9, 0.766, 0.5992, 0.5436, 0.0912]

  middle = 4
  linewidth = 0.5

  for i in range(len(opacities)):
    if i < middle:
      axes.plot(x, y[:,i], color=color, alpha=opacities[i], linewidth=linewidth)
      axes.fill_between(x, y[:,i], y[:,i+1], color=color, alpha=opacities[i])
    if i == middle:
      axes.plot(x, y[:,i], label=label, color=color, alpha=opacities[i], linewidth=linewidth)
    if i > middle:
      axes.plot(x, y[:,i], color=color, alpha=opacities[i], linewidth=linewidth)
      axes.fill_between(x, y[:,i-1], y[:,i], color=color, alpha=opacities[i])


def average_models(x, y):
  mean = np.mean(y, axis=0)
  assert len(x) == len(mean)

  if len(y) == 1:
    lo, hi = None, None
  else:
    lo = np.amin(y, axis=0)
    hi = np.amax(y, axis=0)
    assert len(x) == len(hi)
    assert len(x) == len(lo)

  return dict(mean=mean, lo=lo, hi=hi)


def average_max(y):
  best = np.amax(y, axis=1)
  # return np.mean(best, axis=0), np.std(best, axis=0)
  return np.mean(best, axis=0)


def process_data(args):
  """Group models, read their data and process it"""
  legend  = args.conf["legend"]
  groups  = group_models(legend)
  print("Groups:")
  pprint.pprint(groups)
  print()
  groups  = process_group(groups, args)
  return groups


def plot_histos(args, groups):
  """Create a Figure and plot the data
  Returns:
    Figure: the plotted figure
  """
  legend  = args.conf["legend"]
  envs    = sorted(groups.keys())
  nplots  = sum([len(labels) for env, labels in groups.items()])

  fig, axes = make_figure(args, nplots)

  i = 0
  for env in envs:
    labels = groups[env]
    for label, runs in labels.items():
      color = legend[label]["color"]
      # Only the first entry is plotted for histograms, the rest are ignored
      print("Processing env: '%s', label: '%s'" % (env, label))
      data = runs[0]
      x, y = data.x, data.y

      ax = axes[i]
      ax.set_title(label=env)
      plot_histogram(ax, x, y, label, color, args.linewidth)
      i = i+1

  add_legend(fig, axes, nplots, args.shrink, args.linewidth)

  return fig


def plot_figure(args, groups):
  """Create a Figure and plot the data
  Returns:
    Figure: the plotted figure
  """
  legend  = args.conf["legend"]
  envs    = sorted(groups.keys())
  nplots  = len(envs)

  fig, axes = make_figure(args, nplots)

  # Plot each figure
  for env, ax in zip(envs, axes):
    # Set the title to the environment name
    ax.set_title(label=env)
    labels = groups[env]
    for label, runs in labels.items():
      color = legend[label]["color"]
      x = runs[0].x
      y = [mdata.y for mdata in runs]
      print("Processing env: '%s', label: '%s'" % (env, label))
      y = average_models(x, y)
      plot_model(ax, x, y, label, color, args.linewidth)

  add_legend(fig, axes, nplots, args.shrink, args.linewidth)

  return fig


def benchmark_scores(groups, compute_score=average_max):
  """Read data, process it, create a Figure and plot the data
  Returns:
    dict of the best scores
  """
  envs    = sorted(groups.keys())
  scores  = dict()

  # Traverse all environments
  for env in envs:
    scores[env] = dict()
    labels = groups[env]
    for label, runs in labels.items():
      y = [mdata.y for mdata in runs]
      print("Processing env: '%s', label: '%s'" % (env, label))
      scores[env][label] = compute_score(y)

  return scores


def main():
  # Process args and conf
  args = parse_args()

  groups = process_data(args)

  if args.run == "plot":
    # Set the global fontsize for all plots
    plt.rc('font', size=args.fontsize)

    # Plot figure
    fig = plot_figure(args, groups)
    # fig = plot_histos(args, groups)

    # Save figure
    pngpath = os.path.join(FIG_DIR, args.filename + ".png")
    plt.savefig(pngpath)

    plt.show()

  elif args.run == "log":
    scores  = benchmark_scores(groups)

    txtfile = os.path.join(FIG_DIR, args.filename)
    dataio.save_scores(scores, txtfile, args)


if (__name__ == "__main__"):
  main()
