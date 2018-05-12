import json
import logging
import os
import numpy as np

from gym.utils  import atomic_write
from rltf.utils import seeding


logger = logging.getLogger(__name__)


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
    self.new_idx   = 0

    self.obs    = np.empty([self.max_size] + obs_shape, dtype=obs_dtype)
    self.action = np.empty([self.max_size] + act_shape, dtype=act_dtype)
    self.reward = np.empty([self.max_size],             dtype=np.float32)
    self.done   = np.empty([self.max_size],             dtype=np.bool)

    self.prng   = seeding.get_prng()


  def store(self, obs_t, act_t, reward_tp1, done_tp1):
    raise NotImplementedError()


  def sample(self, batch_size):
    raise NotImplementedError()


  def new_data(self, batch_size=32):
    """Yields the new data which was stored since the last call to this function.
    Args:
      batch_size: int. Size of a single yielded batch. Can be smaller than specified if not enough data
    Returns:
      python generator; has the same signature as `sample()`
    """
    raise NotImplementedError()


  def all_data(self, batch_size=32):
    """Yields all data in the buffer
    Args:
      batch_size: int. Size of a single yielded batch. Can be smaller than specified if not enough data
    Returns:
      python generator which should be iterated; has the same signature as `sample()`
    """
    raise NotImplementedError()


  def recent_data(self, size, batch_size=32):
    """Yields the most recent `size` number of examples in the buffer
    Args:
      size: int. Total number of data points to generate
      batch_size: int. Size of a single yielded batch. Can be smaller than specified if not enough data
    Returns:
      python generator which should be iterated; has the same signature as `sample()`
    """
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
    save_dir    = os.path.join(model_dir, "buffer")
    state_file  = os.path.join(save_dir, "state.json")

    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    np.save(os.path.join(save_dir, "obs.npy"),   self.obs[:self.size_now])
    np.save(os.path.join(save_dir, "act.npy"),   self.action[:self.size_now])
    np.save(os.path.join(save_dir, "rew.npy"),   self.reward[:self.size_now])
    np.save(os.path.join(save_dir, "done.npy"),  self.done[:self.size_now])

    data = {
      "size_now": self.size_now,
      "next_idx": self.next_idx,
      "new_idx":  self.new_idx,
    }

    with atomic_write.atomic_write(state_file) as f:
      json.dump(data, f, indent=4, sort_keys=True)


  def restore(self, model_dir):
    """Populate the buffer from data previously saved to disk
    Args:
      model_dir: Full path of the directory of the data
    """
    save_dir    = os.path.join(model_dir, "buffer")
    state_file  = os.path.join(save_dir, "state.json")

    if not os.path.exists(save_dir):
      return logger.warning("BaseBuffer not saved and cannot resume. Continuing with empty buffer.")

    with open(state_file, 'r') as f:
      data = json.load(f)

    self.size_now = data["size_now"]
    self.next_idx = data["next_idx"]
    self.new_idx  = data["new_idx"]

    obs    = np.load(os.path.join(save_dir, "obs.npy"))
    action = np.load(os.path.join(save_dir, "act.npy"))
    done   = np.load(os.path.join(save_dir, "done.npy"))
    reward = np.load(os.path.join(save_dir, "rew.npy"))

    assert len(obs) == len(action) == len(reward) == len(done) == self.size_now
    assert self.obs.shape[1:]     == obs.shape[1:]
    assert self.action.shape[1:]  == action.shape[1:]
    assert self.reward.shape[1:]  == reward.shape[1:]
    assert self.done.shape[1:]    == done.shape[1:]

    self.obs[:self.size_now]    = obs
    self.action[:self.size_now] = action
    self.reward[:self.size_now] = reward
    self.done[:self.size_now]   = done


  def _sample_n_unique(self, n, lo, hi, exclude=None):
    """Sample n unique indices in the range [lo, hi), making sure no sample appreas in `exclude`
    Args:
      n: int. Number of samples to take
      lo: int. Lower boundary of the sample range; inclusive
      hi: int. Upper boundary of the sample range; exclusive
      exclude: list or np.array. Contains values that samples must not take
    Returns:
      np.array of the sampled indices
    """

    batch = np.empty(n, dtype=np.uint32)
    k = 0

    while k < n:
      samples = self.prng.randint(lo, hi, n-k)
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
