import threading
import numpy as np

from rltf.memory  import BaseBuffer
from rltf.utils   import seeding


class ReplayBuffer(BaseBuffer):
  """Uniform replay buffer. Supports both image observations and low-level
  observations
  """

  def __init__(self, size, state_shape, obs_dtype, act_shape, act_dtype, obs_len=1, sync=False):
    """
    Args: `See BaseBuffer.__init__()`
    """

    super().__init__(size, state_shape, obs_dtype, act_shape, act_dtype, obs_len)

    self._sync    = sync and seeding.SEEDED
    self._sampled = threading.Event()
    self._stored  = threading.Event()
    self._sampled.clear()
    self._stored.set()


  def store(self, obs_t, act_t, rew_tp1, done_tp1):
    """See `BaseBuffer.store()`"""

    self.wait_sampled()

    super().store(obs_t, act_t, rew_tp1, done_tp1)

    self.signal_stored()


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

    inds    = self._sample_n_unique(batch_size, 0, self.size_now, exclude)
    samples = self._batch_samples(inds)

    return samples


  def _batch_samples(self, inds):
    """Takes the samples from the buffer stacks them into a batch
    Args:
      inds: np.array or list. Indices for transitions to be sampled from the buffer
    Returns:
      See self.sample()
    """
    next_inds = (inds+1) % self.max_size
    if self.obs_len == 1:
      obs_batch     = self.obs[inds]
      obs_tp1_batch = self.obs[next_inds]
    else:
      obs_batch     = np.concatenate([self._encode_img_observation(idx)[None] for idx in inds], 0)
      obs_tp1_batch = np.concatenate([self._encode_img_observation(idx)[None] for idx in next_inds], 0)

    act_batch = self.action[inds]
    rew_batch = self.reward[inds]
    done_mask = self.done[inds]

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
    # If another thread might be currently modifying the data in idx and it is not known whether it
    # already incremented it, then the points with inconsistent history must also be incremented by 1.
    # Also the index with invalid next state is either idx-1 (thread has not incremened idx yet) or
    # idx (we have read the incremented idx). In either case, the safe lower bound remains idx-1.
    # If self.sync == True, then `store()` has not begun and the upper bound is idx+obs_len-1
    # NOTE: QlearnAgent can call `store()` only once before `sample()` finishes. If it calls
    # `sample()` twice, before `store()` finishes, nothing changes.

    idx     = self.next_idx
    exclude = np.arange(idx-1, idx+self.obs_len) % self.max_size
    return exclude


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
