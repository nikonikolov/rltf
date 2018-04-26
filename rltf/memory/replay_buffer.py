import threading
import numpy as np

from rltf.memory.base_buffer  import BaseBuffer
from rltf.utils               import seeding


class ReplayBuffer(BaseBuffer):
  """Uniform replay buffer. Supports both image observations and low-level
  observations
  """

  def __init__(self, size, state_shape, obs_dtype, act_shape, act_dtype, obs_len=1, sync=False):
    """
    Args:
      Check super().__init__()
      obs_len: int. The number of observations that comprise one state. Must be `>= 1`.
        If `obs_len=1`, then `obs_shape == state_shape`. If `obs_len>=1`, then observations
        are assumed to be images and `obs_shape[-1] == state_shape[-1] / obs_len`. Corresponds
        to the observation order of the MDP.
    """
    assert obs_len > 0
    assert isinstance(obs_len, int)
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

    super().__init__(size, obs_shape, obs_dtype, act_shape, act_dtype)

    # state_shape: The shape of the state: the shape receieved as observation from the environment
    # and the shape returned by the buffer
    # obs_shape: The shape of a single observation: several observations make one state (k-th order MDP).
    # This is the shape of the data stored in the buffer in order to avoid storing the same values
    # several times
    self.obs_len      = obs_len
    self.obs_shape    = obs_shape
    self.state_shape  = state_shape

    self._sync    = sync and seeding.SEEDED
    self._sampled = threading.Event()
    self._stored  = threading.Event()
    self._sampled.clear()
    self._stored.set()


  def store(self, obs_t, act_t, reward_tp1, done_tp1):
    """
    Store the observed transition defined as: Given `obs_t`, action `act_t` was taken.
    Then reward `reward_tp1` was observed. If after action `act_t` the episode terminated,
    then `done_tp1` will be `True`, otherwise `False`. Note that the observation after taking
    `act_t` should be passed as `obs_t` on the next call to `store()`. NOTE: if `done_tp1 == True`,
    then there is no need to call `store()` on `obs_tp1`: we do NOT need to know it since we never
    use it in computing the backup value
    Args:
      obs_t: `np.array`, `shape=state_shape`. Observation directly from the env
      act_t: `np.array`, `shape=state_shape` or `float`
      reward_tp1: `float`
      done_tp1: `bool`
    """
    self.wait_sampled()

    # To avoid storing the same data several times, if obs_len > 1, then store only the last
    # observation from the stack of observations that comprise a state
    if self.obs_len > 1:
      self.obs[self.next_idx]   = obs_t[:, :, -self.obs_shape[-1]:]
    else:
      self.obs[self.next_idx]   = obs_t

    self.action[self.next_idx]  = act_t
    self.reward[self.next_idx]  = reward_tp1
    self.done[self.next_idx]    = done_tp1

    idx = self.next_idx
    self.next_idx = (self.next_idx + 1) % self.max_size
    self.size_now = min(self.max_size, self.size_now + 1)

    self.signal_stored()

    return idx


  def sample(self, batch_size):
    """
    Sample uniformly `batch_size` different transitions. Note that the
    implementation is thread-safe and allows for another thread to be currently
    adding a new transition to the buffer.

    i-th sample transition is as follows: when state was `obs[i]`, action
    `act[i]` was taken. After that reward `rew[i]` was received and subsequent
    state `obs_tp1[i]` was observed. If `done[i]` is True, then the episode was
    finished after taking `act[i]` and `obs_tp1[i]` will be garbage

    Args:
      batch_size: int. Size of the batch to sample
    Returns:
      Python dictionary with keys
      "obs": np.array, shape=[batch_size, state_shape], dtype=obs_dtype, Batch states
      "act": np.array, shape=[batch_size, act_shape], dtype=act_dtype. Batch actions
      "rew": np.array, shape=[batch_size, 1], dtype=np.float32. Batch rewards
      "obs_tp1": np.array, shape=[batch_size, obs_shape], dtype=obs_dtype. Batch next state
      "done": np.array, shape=[batch_size, 1], dtype=np.bool. Batch done mask.
        True if episode has ended, False otherwise
    """

    self.wait_stored()
    exclude = self._exclude_indices()
    self.signal_sampled()

    assert batch_size < self.size_now - len(exclude) - 1

    idxes   = self._sample_n_unique(batch_size, 0, self.size_now-2, exclude)
    samples = self._batch_samples(idxes)

    return samples


  def _batch_samples(self, idxes):
    """Takes the samples from the buffer stacks them into a batch
    Args:
      idxes: np.array or list. Indices for transitions to be sampled from the buffer
    Returns:
      See self.sample()
    """
    if self.obs_len == 1:
      obs_batch     = self.obs[idxes]
      obs_tp1_batch = self.obs[idxes+1]
    else:
      obs_batch     = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
      obs_tp1_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes+1], 0)

    act_batch     = self.action[idxes]
    rew_batch     = self.reward[idxes]
    done_mask     = self.done[idxes]

    return dict(obs=obs_batch, act=act_batch, rew=rew_batch, obs_tp1=obs_tp1_batch, done=done_mask)


  def _exclude_indices(self):
    """Compute indices that must be excluded because the information there
    might be incosistent or being currently modified.
    Returns:
      list or np.array of indices to exclude
    """

    # Assume no other thread is modifying self.next_idx. Then idx points
    # to the observation that will be overwritten next time. Then:
    # - the idx-1 observation is invalid because the next obs is not the true one
    # - the [idx : idx+obs_len-1) observations have inconsistent history
    # If another thread might be currently incrementing the original value of
    # self.next_idx, then the actual values that idx can have at this
    # point are self.next_idx or self.next_idx+1. Then we need to widen the
    # exlude range by 1 at each end. However, the previous observation is no longer
    # guaranteed to be correctly written, so we need to widen the min exclude
    # range by 1 more

    idx     = self.next_idx
    exclude = np.arange(idx-3, idx+self.obs_len) % self.max_size
    return exclude


  def _encode_observation(self, idx):
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
      # This optimization has potential to saves about 30% compute time \o/
      img_h, img_w = self.obs.shape[1], self.obs.shape[2]
      return self.obs[lo:hi].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)


  def wait_sampled(self):
    if not self._sync:
      return
    # Wait until an action is chosen to be run
    while not self._sampled.is_set():
      self._sampled.wait()
    self._sampled.clear()

  def wait_stored(self):
    if not self._sync:
      return
    # Wait until training step is done
    while not self._stored.is_set():
      self._stored.wait()
    self._stored.clear()

  def signal_sampled(self):
    if not self._sync:
      return
    # Signal that the action is chosen and the TF graph is safe to be run
    self._sampled.set()

  def signal_stored(self):
    if not self._sync:
      return
    # Signal to env thread that the training step is done running
    self._stored.set()
