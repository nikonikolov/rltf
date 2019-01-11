import yaml
import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tabulate


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
    data = [scores[env].get(label, -float("inf")) for label in labels]
    csvdata.append([env] + data)
    if args.boldmax:
      best = max(data)
      data = ["{:,.1f}".format(score) if score != best else "\\textbf{{{:,.1f}}}".format(score) for score in data]
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
  if not os.path.exists(file):
    raise ValueError("Configuration file {} does not exist".format(file))
  with open(file, 'r') as f:
    conf = yaml.load(f)

  assert "legend" in conf
  assert "root_dir" in conf
  assert os.path.exists(conf["root_dir"])
  for label, props in conf["legend"].items():
    assert "models" in props
    assert "color" in props

  return conf


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

  # import tensorflow as tf
  writer = tf.summary.FileWriter(tb_dir)

  for i, s in enumerate(steps):
    summary = tf.Summary()
    for tag, vals in data.items():
      summary.value.add(tag=tag, simple_value=vals[i])
    writer.add_summary(summary, global_step=s)

  writer.flush()
  writer.close()
