import json
import os
from collections import OrderedDict

import numpy as np
import tabulate

CODE_DIR   = os.path.abspath(os.path.dirname(__file__))
CONF_DIR   = os.path.join(CODE_DIR, "conf")

def save_scores(scores, file, args):
  """Write scores in table format to a .txt file and to a .tex file (in latex format)
  Args:
    scores: dict
    file: str. Does not contain the extension
    args: ArgumentParser. The command-line arguments
  """
  envs    = sorted(scores.keys())
  labels  = [label for label in args.conf["legend"]]
  csvdata = []
  texdata = []

  for env in envs:
    csvdata.append([env] + [scores[env].get(label, -float("inf")) for label in labels])
    if args.tablemax:
      data = [scores[env].get(label, -float("inf")) for label in labels]
      best = max(data)
      data = ["%.1f" % score if score != best else "\\textbf{%.1f}" % score for score in data]
      texdata.append([env] + data)

  csvtable = tabulate.tabulate(csvdata, headers=labels, floatfmt=".1f", tablefmt="presto")
  textable = tabulate.tabulate(texdata, headers=labels, floatfmt=".1f", tablefmt="latex_raw")

  with open(file + ".txt", 'w') as f:
    f.write(csvtable)

  with open(file + ".tex", 'w') as f:
    f.write(textable)


def get_model_props(conf, model):
  props = conf["legend"][model]
  return props["label"], props["color"]


def get_model_name(model_dir):
  s    = model_dir.find("/")
  name = model_dir[:s]
  return name


def get_env_name(model_dir):
  """
  Args:
    model_dir: str. Will be in the format model-name/env-name_run-date and might end in "/"
  Return:
    str with the env name as it appears in gym
  """
  len_date = 20
  if model_dir[-1] == "/":
    len_date += 1
  env = model_dir[:-20]
  s   = env.find("/")
  env = env[s+1:]
  s = env.find("NoFrameskip")
  if s > 0:
    env = env[:s]
  else:
    s = env.find("-v")
    env = env[:s]
  return env


def get_model_dir(model, args):
  return os.path.join(args.conf["root_dir"], model)


def read_conf(file):
  file = os.path.join(CONF_DIR, file)
  if not os.path.exists(file):
    raise ValueError("Configuration file does not exist")
  with open(file, 'r') as f:
    # conf = json.load(f)
    conf = json.load(f, object_pairs_hook=OrderedDict)

  assert "legend" in conf
  assert "root_dir" in conf
  assert os.path.exists(conf["root_dir"])
  for label, props in conf["legend"].items():
    assert "models" in props
    assert "color" in props

  return conf


def read_npy(file):
  if os.path.exists(file):
    return np.load(file)
  return None


def read_model_data(model_dir):
  assert os.path.exists(model_dir)

  ep_lens = read_npy(os.path.join(model_dir, "env_monitor/data/eval_ep_lens.npy"))
  ep_rews = read_npy(os.path.join(model_dir, "env_monitor/data/eval_ep_rews.npy"))

  scores_steps = read_npy(os.path.join(model_dir, "env_monitor/data/eval_scores_steps.npy"))
  scores_inds  = read_npy(os.path.join(model_dir, "env_monitor/data/eval_scores_inds.npy"))

  if scores_steps is not None and scores_inds is not None:
    assert len(scores_steps) == len(scores_inds)
    assert ep_lens is not None
    assert ep_rews is not None

  return dict(
              scores_steps=scores_steps,
              scores_inds=scores_inds,
              ep_rews=ep_rews,
              ep_lens=ep_lens,
             )


# TODO: Allow for reading histograms
def read_tb_file(model_dir, tag):
  """Read data from a tensorboard file
  Args:
    model_dir: str. Just the model directory path, without the TB addition
    tag: str. Tag which to fetch. Must start with train/ or eval/
  """
  import tensorflow as tf

  tb_dir = os.path.join(model_dir, "tf/tb")
  files  = os.listdir(tb_dir)
  files  = [os.path.join(tb_dir, file) for file in files]
  x, y   = [], []

  # Check tag for correctness
  assert tag.startswith("train/") or tag.startswith("eval/") or tag.startswith("debug/")

  # Read TB file
  for file in files:
    # Traverse events/summaries
    for e in tf.train.summary_iterator(file):
      # Traverse every value in the summary
      for v in e.summary.value:
        if tag in v.tag:
          x.append(e.step)
          y.append(v.simple_value)

  # Sort the lists by step
  x, y = (list(t) for t in zip(*sorted(zip(x, y), key=lambda t: t[0])))
  return x, y


def write_tb_file(tb_dir, steps, data):
  """
  Args:
    tb_dir: str. Directory where the file should be opened
    steps: list. List of the event time steps
    data: dict. Every key is a tag and every value is a list of the data for the tag. The length of
      the list must equal the length of steps
  """
  # Check for correctness
  for tag, vals in data.items():
    assert tag.startswith("train/") or tag.startswith("eval/")
    assert len(steps) == len(vals)

  import tensorflow as tf
  writer = tf.summary.FileWriter(tb_dir)

  for i, s in enumerate(steps):
    summary = tf.Summary()
    for tag, vals in data.items():
      summary.value.add(tag=tag, simple_value=vals[i])
    writer.add_summary(summary, global_step=s)

  writer.flush()
  writer.close()
