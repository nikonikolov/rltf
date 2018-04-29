from collections import OrderedDict

ids_layout = {
  "width": 500,
  "height": 260,
  "video_top": 25,
  "video_left": 0,
  "figures": {
    "actions": {
      "top":20,
      "left":180,
      "fig": {
        "width": 320,
        "height": 240,
        "subplots": dict(nrows=3, ncols=1, sharex=True),
        "subplots_conf": OrderedDict(
          a_mean={
            "set_title": dict(label="MEAN", size=6),
          },
          a_std={
            "set_title": dict(label="STD", size=6),
          },
          a_ids={
            "set_title": dict(label="IDS", size=6),
            "set_xticklabels": dict(labels=['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'], 
                                    size=5.5),
          },
        ),
        "subplots_common": {
          # "grid": dict(),   # Args for Axes.grid()
          "tick_params": dict(axis='y', labelsize=6),
        },
        "fig_conf": {
          "tight_layout": dict(pad=1.0, h_pad=0.0),
        },
      },
      "plot": {
        "method": "bar",
        "kwargs": dict(x=['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'], color='#1f77b4'),
        "keep": False,
      },
    },
  }
}
