import json
import logging
import os
import numpy as np

from gym.utils  import atomic_write
from rltf.utils import rltf_conf
from rltf.utils import seeding


logger = logging.getLogger(__name__)


class BaseBuffer():
  """Abstract buffer that saves agent experience. Supports both image and low-dimensional observations.
  Very memory efficient implementation in the case of images."""

  def __init__(self, size, state_shape, obs_dtype, act_shape, act_dtype, obs_len):
    """
    Args:
      state_shape: tuple or list. Shape of what is consedered to be a single state (not observation).
        For example, for DQN this should be `[84, 84, 4]` because a state is comprised of the last 4
        frames (observations).
      obs_dtype: np.dtype. Type of the observation data
      act_shape: tuple or list. Shape of the action space
      act_dtype: np.dtype. Type of the action data
      obs_len: int, `>= 1`. The number of observations that comprise a state. If `obs_len=1`,
        then `obs_shape == state_shape`. Must equal 1 for low-dimensional observations.
        If `obs_len>=1`, then observations must be images. In this case, states are comprised of
        stacked consecutive observations (images) and `obs_shape[-1] == state_shape[-1] / obs_len`.
        In this case the buffer stores observations separately and automatically reconstructs the
        full states when queried. Corresponds to the order of the MDP.
    """

    # Compute the observation shape
    obs_shape       = self._get_obs_shape(state_shape, obs_len, obs_dtype)

    self.obs_shape  = list(obs_shape)   # observation shape (NOT state shape!)
    self.act_shape  = list(act_shape)
    self.obs_len    = obs_len

    self.max_size   = int(size)
    self.size_now   = 0
    self.next_idx   = 0
    # self.new_idx    = 0

    # Create the buffers
    self.obs    = np.empty([self.max_size] + self.obs_shape,  dtype=obs_dtype)
    self.action = np.empty([self.max_size] + self.act_shape,  dtype=act_dtype)
    self.reward = np.empty([self.max_size],                   dtype=np.float32)
    self.done   = np.empty([self.max_size],                   dtype=np.bool)

    self.prng   = seeding.get_prng()


  @staticmethod
  def _get_obs_shape(state_shape, obs_len, obs_dtype):
    """Compute the shape of a single observation (not state)"""

    assert isinstance(obs_len, int) and obs_len >= 1

    # Only image observations support stacking observations
    if obs_len > 1:
      assert len(state_shape) == 3
    # Make sure that the type of the observation is np.uint8 for images
    if len(state_shape) == 3:
      assert obs_dtype == np.uint8

    # Images assume that the last dimension of the shape is the channel dimension
    if obs_len > 1 and len(state_shape) == 3:
      assert state_shape[-1] % obs_len == 0
      obs_shape = list(state_shape)
      obs_shape[-1] = int(obs_shape[-1]/obs_len)
    else:
      obs_shape = state_shape

    return obs_shape


  def store(self, obs_t, act_t, rew_tp1, done_tp1):
    """Store an observed transition. If `obs_len>1`, the next call to this function must be with
    the observation after taking `act_t`, otherwise, reconstructed state will be incorrect.
    If `done_tp1 == True`, then `store()` should not be called with `obs_tp1`, since the agent
    does not need it for computing the return
    Args:
      obs_t: `np.array`, of shape `state_shape`. If `obs_len>1`, the observation is automatically
        extracted and stored instead of storing duplicate data.
      act_t: `np.array`, of shape `act_shape` or `float`. Action taken when `obs_t` was observed
      reward_tp1: `float`. Reward obtained on executing `act_t` in state `obs_t`
      done_tp1: `bool`. True if episode terminated on executing `act_t` in state `obs_t`.
    """

    # To avoid storing the same data several times, if obs_len > 1, then store only the last
    # observation from the stack of observations that comprise a state
    if self.obs_len > 1:
      self.obs[self.next_idx]   = obs_t[:, :, -self.obs_shape[-1]:]
    else:
      self.obs[self.next_idx]   = obs_t

    self.action[self.next_idx]  = act_t
    self.reward[self.next_idx]  = rew_tp1
    self.done[self.next_idx]    = done_tp1

    self.next_idx = (self.next_idx + 1) % self.max_size
    self.size_now = min(self.max_size, self.size_now + 1)


  def _encode_img_observation(self, idx):
    """Encode the observation for idx by stacking the `obs_len` preceding frames together.
    Assume there are more than `obs_len` frames in the buffer.
    NOTE: Used only for image observations
    """
    hi = idx + 1 # make noninclusive
    lo = hi - self.obs_len

    for i in range(lo, hi - 1):
      if self.done[i % self.max_size]:
        lo = i + 1
    missing = self.obs_len - (hi - lo)

    # We need to duplicate the lo observation
    if missing > 0:
      frames = [self.obs[lo % self.max_size] for _ in range(missing)]
      for i in range(lo, hi):
        frames.append(self.obs[i % self.max_size])
      return np.concatenate(frames, 2)
    # We are on the boundary of the buffer
    elif lo < 0:
      img_h, img_w = self.obs.shape[1], self.obs.shape[2]
      frames = [self.obs[lo:], self.obs[:hi]]
      frames = np.concatenate(frames, 0)
      return frames.transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)
    # The standard case
    else:
      # This optimization can save about 30% compute time
      img_h, img_w = self.obs.shape[1], self.obs.shape[2]
      return self.obs[lo:hi].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)


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


  def save(self, model_dir):
    """Store the data to disk
    Args:
      model_dir: Full path of the directory to save the buffer
    """
    save_dir    = os.path.join(model_dir, "buffer")
    state_file  = os.path.join(save_dir, "state.json")

    if not os.path.exists(save_dir):
      # Create symlink to store buffer if $RLTFBUF is defined
      if 'RLTFBUF' in os.environ:
        # split     = os.path.split(os.path.normpath(model_dir))
        # envdir    = split[1]
        # model     = os.path.split(split[0])
        # store_dir = os.path.join(os.environ['RLTFBUF'], os.path.join(model, envdir))
        mdir      = os.path.relpath(model_dir, rltf_conf.MODELS_DIR)
        store_dir = os.path.join(os.environ['RLTFBUF'], mdir)

        store_dir = os.path.join(store_dir, "buffer")
        if not os.path.exists(store_dir):
          os.makedirs(store_dir)
        os.symlink(store_dir, save_dir)
      # Store the buffer directly in the folder
      else:
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, "obs.npy"),   self.obs[:self.size_now])
    np.save(os.path.join(save_dir, "act.npy"),   self.action[:self.size_now])
    np.save(os.path.join(save_dir, "rew.npy"),   self.reward[:self.size_now])
    np.save(os.path.join(save_dir, "done.npy"),  self.done[:self.size_now])

    data = {
      "size_now": self.size_now,
      "next_idx": self.next_idx,
      # "new_idx":  self.new_idx,
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
    # self.new_idx  = data["new_idx"]

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


  def reset(self):
    self.size_now   = 0
    self.next_idx   = 0
    # self.new_idx    = 0


  @property
  def size(self):
    return self.max_size


  def __len__(self):
    return self.size_now
