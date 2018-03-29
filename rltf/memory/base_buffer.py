import os

import numpy as np


class BaseBuffer():

  def __init__(self, size, obs_shape, obs_dtype, act_shape, act_dtype):
    """
    Args:
      obs_shape: tuple or list. Shape of a single observation
      obs_dtype: np.dtype. Type of the observation data
      act_shape: tuple or list. Shape of the action space
      act_dtype: np.dtype. Type of the action data
    """

    obs_shape   = list(obs_shape)
    act_shape   = list(act_shape)

    self.max_size  = int(size)
    self.size_now  = 0
    self.next_idx  = 0

    self.obs    = np.empty([self.max_size] + obs_shape, dtype=obs_dtype)
    self.action = np.empty([self.max_size] + act_shape, dtype=act_dtype)
    self.reward = np.empty([self.max_size],             dtype=np.float32)
    self.done   = np.empty([self.max_size],             dtype=np.bool)


  def store(self, obs_t, act_t, reward_tp1, done_tp1):
    raise NotImplementedError()


  def sample(self, batch_size):
    raise NotImplementedError()


  @property
  def size(self):
    return self.max_size


  def __len__(self):
    return self.size_now


  def save(self, model_dir, obj_name=None):
    """Store the data to disk

    NOTE: Undefined behavior if another thread accesses the object

    Args:
      model_dir: Full path of the directory to save the buffer
      obj_name: Name of the object - used to uniquely identify the created files
    """

    if obj_name is None:
      obj_name = self.__class__.__name__

    np.save(os.path.join(model_dir, obj_name + "_obs.npy"),   self.obs)
    np.save(os.path.join(model_dir, obj_name + "_act.npy"),   self.action)
    np.save(os.path.join(model_dir, obj_name + "_done.npy"),  self.done)
    np.save(os.path.join(model_dir, obj_name + "_rew.npy"),   self.reward)

    obs     = self.obs
    action  = self.action
    reward  = self.reward
    done    = self.done
    self.obs    = None
    self.action = None
    self.reward = None
    self.done   = None

    pickle_save(os.path.join(model_dir, obj_name + ".pkl"), self)

    self.obs    = obs
    self.action = action
    self.reward = reward
    self.done   = done


  def restore(self, model_dir, obj_name):
    self        = pickle_restore(os.path.join(model_dir, obj_name + ".pkl"))
    self.obs    = np.load(os.path.join(model_dir, obj_name + "_obs.npy"))
    self.action = np.load(os.path.join(model_dir, obj_name + "_act.npy"))
    self.done   = np.load(os.path.join(model_dir, obj_name + "_done.npy"))
    self.reward = np.load(os.path.join(model_dir, obj_name + "_rew.npy"))

    return self


  def _sample_n_unique(self, n, low, high, exclude=None):
    """Sample n unique indices in the range [low, high) and
    while making sure that no sample appreas in exclude

    Args:
      n: int. Number of samples to take
      low: int. Lower boundary of the sample range
      high: int. Upper boundary of the sample range
      exclude: list or np.array. Contains values that samples must not take
    """

    batch = np.empty(n, dtype=np.uint32)
    if exclude is not None:
      exclude = np.asarray(exclude)
    k = 0

    while k < n:
      samples = np.random.randint(low, high, n-k)
      # Get only the unique entries
      samples = np.unique(samples)
      # Get only the entries which are not in exclude
      if exclude is not None:
        valid = np.all(samples[:, None] != exclude, axis=-1)
        samples = samples[valid]
      # Update batch
      end = min(k + samples.shape[0], n)
      batch[k:end] = samples
      k = end

    return batch
