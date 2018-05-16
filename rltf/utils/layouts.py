from collections import OrderedDict

def plot_bars(ax, kwargs, env, color):
  x = atari_labels(env.unwrapped.get_action_meanings())
  return ax.bar(x=x, **kwargs, color=color)


def plot_highlight_bars(ax, kwargs, env, color_n='#1f77b4', color_hi='#d62728'):
  x = atari_labels(env.unwrapped.get_action_meanings())
  color = [color_n] * len(x)
  a = kwargs.pop("a")
  color[a] = color_hi
  return ax.bar(x=x, **kwargs, color=color)


def atari_labels(x):
  for i, label in enumerate(x):

    if label[-4:] == "FIRE":
      if len(label) > 4:
        end = "\nFIRE"

        length = len(label[:-4])
        if length >= 6:
          if label[:2] == "UP":
            start = "UP\n" + label[2:-4]
          elif label[:4] == "DOWN":
            start = "DOWN\n" + label[4:-4]
          else:
            raise ValueError
        else:
          start = label[:-4]
        x[i] = start + end

    elif len(label) >= 6:
      length = len(label)
      if label[:2] == "UP":
        x[i] = "UP\n" + label[2:]
      elif label[:4] == "DOWN":
        x[i] = "DOWN\n" + label[4:]
      else:
        raise ValueError

  return x


ids_layout = {
  "width": 800,
  "height": 300,
  "obs_align": dict(vertical='center', horizontal='left'),
  # "obs_scale": 1.0,
  "figures": {
    "train_actions": {
      "align": dict(vertical='center', horizontal='right'),
      "width": 620,
      "height": -1,
      "fig": {
        "subplots": dict(nrows=3, ncols=1, sharex=True),
        "subplots_conf": OrderedDict(
          a_mean={
            "tick_params": dict(axis='y', labelsize=5.5),
            "set_title": dict(label="MEAN", size=6),
          },
          a_std={
            "tick_params": dict(axis='y', labelsize=5.5),
            "set_title": dict(label="STD", size=6),
          },
          a_ids={
            "tick_params": dict(axis='y', labelsize=5.5),
            "set_title": dict(label="IDS", size=6),
          },
        ),
        "subplots_common": {
          "grid": dict(linewidth=0.2),
          "tick_params": dict(axis='x', labelsize=6.5),
        },
        "fig_conf": {
          "tight_layout": dict(pad=1.0, h_pad=0.0),
        },
      },
      "plot_function": plot_highlight_bars,
    },
    "eval_actions": {
      "align": dict(vertical='center', horizontal='right'),
      "width": 620,
      "height": -1,
      "fig": {
        "subplots": dict(nrows=1, ncols=1),
        "subplots_conf": OrderedDict(
          a_mean={
            "set_title": dict(label="MEANS", size=8),
            "tick_params": dict(axis='y', labelsize=8),
          },
          # a_vote={
          #   "set_title": dict(label="VOTES", size=8),
          #   "tick_params": dict(axis='y', labelsize=8),
          # },
        ),
        "subplots_common": {
          "tick_params": dict(axis='x', labelsize=6.5),
        },
        "fig_conf": {
          "tight_layout": dict(pad=1.0, h_pad=0.0),
        },
      },
      "plot_function": plot_highlight_bars,
    },
  }
}


qrids_layout = {
  "width": 840,
  "height": 440,
  "obs_align": dict(vertical='center', horizontal='left'),
  # "obs_scale": 1.0,
  "figures": {
    "train_actions": {
      "align": dict(vertical='center', horizontal='right'),
      "width": 660,
      "height": -1,
      "fig": {
        "subplots": dict(nrows=4, ncols=1, sharex=True),
        "subplots_conf": OrderedDict(
          a_mean={
            "tick_params": dict(axis='y', labelsize=5.5),
            "set_title": dict(label="MEAN", size=6),
          },
          a_std={
            "tick_params": dict(axis='y', labelsize=5.5),
            "set_title": dict(label="STD", size=6),
          },
          a_rho2={
            "tick_params": dict(axis='y', labelsize=5.5),
            "set_title": dict(label="RHO2", size=6),
          },
          a_ids={
            "tick_params": dict(axis='y', labelsize=5.5),
            "set_title": dict(label="IDS", size=6),
          },
        ),
        "subplots_common": {
          "grid": dict(linewidth=0.2),
          "tick_params": dict(axis='x', labelsize=6.5),
        },
        "fig_conf": {
          "tight_layout": dict(pad=1.0, h_pad=0.0),
        },
      },
      "plot_function": plot_highlight_bars,
    },
    "eval_actions": {
      "align": dict(vertical='center', horizontal='right'),
      "width": 660,
      "height": -1,
      "fig": {
        "subplots": dict(nrows=1, ncols=1),
        "subplots_conf": OrderedDict(
          # a_mean={
          #   "set_title": dict(label="MEANS", size=8),
          #   "tick_params": dict(axis='y', labelsize=8),
          # },
          a_vote={
            "set_title": dict(label="VOTES", size=8),
            "tick_params": dict(axis='y', labelsize=8),
          },
        ),
        "subplots_common": {
          "tick_params": dict(axis='x', labelsize=6.5),
        },
        "fig_conf": {
          "tight_layout": dict(pad=1.0, h_pad=0.0),
        },
      },
      "plot_function": plot_highlight_bars,
    },
  }
}


layouts = {
  "BstrapDQN_IDS": ids_layout,
  "BstrapQRDQN_IDS": qrids_layout,
}
