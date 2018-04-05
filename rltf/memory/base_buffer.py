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


  def save(self, model_dir):
    """Store the data to disk
    Args:
      model_dir: Full path of the directory to save the buffer
    """
    name = self.__class__.__name__

    np.save(os.path.join(model_dir, name + "_obs.npy"),   self.obs[:self.size_now])
    np.save(os.path.join(model_dir, name + "_act.npy"),   self.action[:self.size_now])
    np.save(os.path.join(model_dir, name + "_rew.npy"),   self.reward[:self.size_now])
    np.save(os.path.join(model_dir, name + "_done.npy"),  self.done[:self.size_now])


  def resume(self, model_dir):
    """Populate the buffer from data previously saved to disk
    Args:
      model_dir: Full path of the directory of the data
    """
    name = self.__class__.__name__

    obs    = np.load(os.path.join(model_dir, name + "_obs.npy"))
    action = np.load(os.path.join(model_dir, name + "_act.npy"))
    done   = np.load(os.path.join(model_dir, name + "_done.npy"))
    reward = np.load(os.path.join(model_dir, name + "_rew.npy"))

    assert len(obs) == len(action) == len(reward) == len(done)
    assert self.obs.shape[1:]     == obs.shape[1:]
    assert self.action.shape[1:]  == action.shape[1:]
    assert self.reward.shape[1:]  == reward.shape[1:]
    assert self.done.shape[1:]    == done.shape[1:]

    self.size_now = len(obs)
    self.next_idx = self.size_now % self.max_size

    self.obs[:self.size_now]    = obs
    self.action[:self.size_now] = action
    self.reward[:self.size_now] = reward
    self.done[:self.size_now]   = done


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
