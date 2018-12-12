import os
import warnings
import numpy as np
import tensorflow as tf

from tensorboard.plugins.distribution.compressor import compress_histogram_proto
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_npy(file):
  if os.path.exists(file):
    return np.load(file)
  return None

class CurveData:
  """Class which contains and manipulates **raw** data. Has no knowledge about where the
  data comes from, just provides a standardized interface for manipulating it.

  Operates on two different structures:
    1) x and y are the same length, i is None; Entries in x correspond to entries in y
    2) x and i are the same length, y is of (maybe) different length; Values in i correspond
       to indices in y: For an index j, at step `x[j]`, the data `y[:i[j]]` was available
  """

  def __init__(self, x, y, i):
    assert x is not None
    assert y is not None

    if i is not None:
      assert len(x) == len(i)
      self.i = np.asarray(i, dtype=np.int32)    # indices for `y`; used to compute y-axis values
    else:
      assert len(x) == len(y)
      # i = [j+1 for j, _ in enumerate(y)]
      self.i = None

    self.x = np.asarray(x, dtype=np.int32)    # x-axis values
    self.y = np.asarray(y, dtype=np.float32)  # y-axis data. NOT necessarily the same length as `x`


  def filter(self, period):
    """Filter x and y such that all entries in x are increments of period
    Args:
      data: Data.
    Returns:
      tuple (x, y)
    """
    assert self.i is None, "Not safe to filter data which depends on indices."
    "Instead, first compute the y-values to eliminate the indices, then filter the data"

    inds  = self.x % period == 0
    steps = self.x[inds]
    vals  = self.y[inds]

    assert steps[1:] - steps[:-1] == period, "Missing step in log"

    # Check for correct parsing
    assert len(vals) != 0 and len(steps) != 0, "Filtering incorrect:"
    "len(x)=%d, len(y)=%d\nx: %s\ny: %s" % (len(steps), len(vals), steps, vals)

    self.x = np.asarray(steps, dtype=np.int32)
    self.y = np.asarray(vals,  dtype=np.float32)


  def set_length(self, max_step, model_name):
    """Limit data up to max_step. If max_step not reached, append the correct step values and -inf
    Args:
      max_step: int. Remove any data after this step
      model_name: str. Name of the model (i.e. model_dir)
    """

    if self.x[-1] == max_step:
      i = len(self.x)-1
    else:
      i = np.searchsorted(self.x, max_step, side='left')

    # max_step not reached; append garbage data
    if i == len(self.x):
      warnings.warn("The recorded data for model '%s' has not reached max_step=%d. "
                    "Appending -inf values" % (model_name, max_step))

      # Get the data logging period in order to append the values
      period = self.x[-1] - self.x[-2]

      x = np.arange(self.x[-1], max_step, period) + period
      y = np.ones([x.shape[0]] + list(self.y.shape[1:]), dtype=np.float32) * -np.inf
      # y = np.ones_like(x, dtype=np.float32) * -np.inf
      # if self.y.ndim == 1:
      #   y = np.ones_like(x, dtype=np.float32) * -np.inf
      # else:
      #   y = np.ones([x.shape[0]] + list(self.y.shape[1:]), dtype=np.float32) * -np.inf

      self.x = np.concatenate([self.x, np.asarray(x, dtype=np.int32)])
      self.y = np.concatenate([self.y, np.asarray(y, dtype=np.float32)])

      if self.i is not None:
        i       = np.arange(0, len(x)) + 1 + self.i[-1]
        self.i  = np.concatenate([self.i, np.asarray(i, dtype=np.int32)])

    # max_step reached; cutoff data
    else:
      i = i+1

      self.x = self.x[:i]

      if self.i is not None:
        self.i = self.i[:i]
        self.y = self.y[:self.i[-1]]
      else:
        self.y = self.y[:i]


  def compute_y(self, mode):
    """Compute the y-values for the curve
    Args:
      mode: str or int.
        If str, must be equal to "mean_score": compute mean between any two steps in x
        If int, indicates a window: compute mean over the last mode values
    """
    if self.i is None:
      return

    # Extract the values between every two entries in self.i;
    # For example, extract episodes from the same evaluation run
    if mode == "mean_score":
      y = [self.y[lo:hi] for lo, hi in zip(np.concatenate([[0],self.i]), self.i)]
      # y = [self.y[lo:hi] for lo, hi in zip([0]+self.i, self.i)]
    # Extract windows of size n
    # For example, extract the last 100 episodes for any step x
    else:
      try:
        n = int(mode)
      except ValueError:
        raise ValueError("Unknown --score-mode")
      y = [self.y[i-n:i] for i in self.i]

    # Average over the extracted data
    y = [np.mean(data) if len(data) > 0 else -np.inf for data in y]
    y = np.asarray(y, dtype=np.float32)
    assert len(y) > 0

    self.y = y
    self.i = None # Indicate that data has been processed; Indices no longer needed


  def smooth_y(self, v):
    """
    Args:
      v: float or int. Smoothing factor
    """

    # Compute running average
    if isinstance(v, float):
      assert v <= 1.0 and v >= 0.0
      # See TF code
      raise NotImplementedError

    # Compute windowed average
    elif isinstance(v, int):
      y = [np.mean(self.y[i-v:i], axis=0) for i in range(1, len(self.y)+1)]
    else:
      raise ValueError("Unknown smoothing factor")

    self.y = np.asarray(y, dtype=np.float32)

# NOT ALLOWED/IMPLEMENTED:
# - Using TB and NP data simultaneously (i.e. falling back to one if the other does not exist)
# - Processing eval and train data simultaneously
# - Reading NP train data - no runs exist which have the steps !!! - need to convert them manually ...

# TODO:
# - Allow for scaling train axis (e.g. in terms of frames)
# - For np train data, just create a function which approximately computes the values ...
# - Forbid accessing data without the getter

class DataWrapper:

  def __init__(self, model_path, max_step, tb_tag=None, data_type=None, log_period=None):
    assert os.path.exists(model_path)

    if tb_tag is not None:
      assert data_type is None
      if tb_tag.startswith("train/") or tb_tag.startswith("debug/"):
        data_type = "t"
      elif tb_tag.startswith("eval/"):
        data_type = "e"
      else:
        raise ValueError("Invalid TB tag. Must start with either 'train/', 'debug/' or 'eval/'")
    else:
      assert data_type is not None
      assert data_type in ["t", "e"]

    # Data-related members
    self.data_type  = data_type
    self.tb_tag     = tb_tag
    self._data      = None

    self.log_period = log_period

    self.model_path = model_path
    self.model_name = os.path.split(os.path.normpath(model_path))[1]

    self._max_train_step = max_step     # Maximum **train** step to consider when computing scores


  @property
  def max_step(self):
    """Get the max step for the raw data
    NOTE: Both in TB and NP, evaluation data is in terms of evaluation steps !!!
    """
    return self._max_train_step


  @property
  def period(self):
    return self.log_period

    # If no log_period provided on command line or not yet read, read the value from file
    if self.log_period is None:

      if self.data_type == 't':
        file = os.path.join(self.model_path, "env_monitor/data/train_stats_summary.json")
      else:
        file = os.path.join(self.model_path, "env_monitor/data/eval_stats_summary.json")

      with open(file, 'r') as f:
        data = json.load(f)

      log_period = data.get("log_period", None)
      assert log_period is not None, "'log_period' not saved by Monitor. "
        "You must provide correct value as argument"

      self.log_period = log_period

    return self.log_period


  @property
  def data(self):
    return self._data


  # def get_data_copy(self, step_mode="t"):
  def get_data(self, step_mode="t"):
    assert step_mode in ["t", "e"]
    assert self._data.i is None, "Data must be processed before accessed"

    return CurveData(x=self._data.x, y=self._data.y, i=None)


  def read_data(self):
    """Read data, limit it to the maximum step and filter from duplicate/too frequent logs"""

    # Read data from TensorBoard
    if self.tb_tag is not None:
      data = self._read_tb_data()
      data = CurveData(**data)
      data.filter(period=self.period)
    # Read data from disk
    else:
      data = self._read_np_data()
      data = CurveData(**data)
      # Cannot filter NP data before computing the values (because of episode indices) !!!

    # Set the data to match max_step in length
    data.set_length(max_step=self.max_step, model_name=self.model_name)

    self._data = data


  def _read_np_data(self):
    if self.data_type == "t":
      x = read_npy(os.path.join(self.model_path, "env_monitor/data/train_log_steps.npy"))
      y = read_npy(os.path.join(self.model_path, "env_monitor/data/train_ep_rews.npy"))
      # i = read_npy(os.path.join(self.model_path, "env_monitor/data/train_ep_lens.npy"))
      i = read_npy(os.path.join(self.model_path, "env_monitor/data/train_log_inds.npy"))

    else:
      x = read_npy(os.path.join(self.model_path, "env_monitor/data/eval_scores_steps.npy"))
      y = read_npy(os.path.join(self.model_path, "env_monitor/data/eval_ep_rews.npy"))
      i = read_npy(os.path.join(self.model_path, "env_monitor/data/eval_scores_inds.npy"))

    assert x is not None
    assert y is not None
    assert i is not None

    return dict(x=x, y=y, i=i)


  def _read_tb_data(self):
    """Read data from a tensorboard file"""

    files = self._get_tb_files()
    x, y   = [], []

    scalar = None

    # Read TB file
    for file in files:
      # Traverse events/summaries
      for e in tf.train.summary_iterator(file):
        # Traverse every value in the summary
        for v in e.summary.value:
          if self.tb_tag == v.tag:
            if scalar is None:
              if v.HasField("simple_value"):
                scalar = True
              elif v.HasField("histo"):
                scalar = False
              else:
                raise ValueError("Only simple_value and histo data can be processed from TB file")

            x.append(e.step)
            if scalar:
              y.append(v.simple_value)
            else:
              # Convert to compressed histogram
              y.append([chv.value for chv in compress_histogram_proto(v.histo)])

    # Check for correct parsing
    assert len(x) > 0 and len(y) > 0, "Parsing TB incorrect:"
    "len(x)=%d, len(y)=%d\nx: %s\ny: %s" % (len(x), len(y), x, y)

    # Sort the lists by step
    x, y = (list(t) for t in zip(*sorted(zip(x, y), key=lambda t: t[0])))

    return dict(x=x, y=y, i=None)


  def _get_tb_files(self):
    # Get the TB directory
    tb_dir = os.path.join(self.model_path, "monitor/tb")
    # Check if the model follows the old directory structure
    if not os.path.exists(tb_dir):
      tb_dir = os.path.join(self.model_path, "tf/tb")

    files  = os.listdir(tb_dir)
    if self.data_type == "t":
      files = [file for file in files if file.endswith("train")]
    else:
      files = [file for file in files if file.endswith("eval")]
    files  = [os.path.join(tb_dir, file) for file in files]
    assert len(files) > 0

    return files
