import numpy as np

from rltf.memory.base_buffer import BaseBuffer


class ReplayBuffer(BaseBuffer):
  """Uniform replay buffer. Supports both image observations and low-level
  observations
  """

  def __init__(self, size, obs_shape, obs_dtype, act_shape, act_dtype, state_len=1):
    """
    Args:
      Check super().__init__()
      state_len: int. Number of observations that comprise one state
    """
    assert state_len > 0
    assert isinstance(state_len, int)
    # Only image observations support stacking observations
    if state_len > 1: 
      assert len(obs_shape) == 3
    # Make sure that the type of the observation is np.uint8 for images
    if len(obs_shape) == 3:
      assert obs_dtype == np.uint8

    super().__init__(size, obs_shape, obs_dtype, act_shape, act_dtype)
    self.state_len = state_len


  def store_frame(self, frame):
    """Store a single frame in the buffer at the next available index, overwriting
    old frames if necessary.

    Parameters
    ----------
    frame: np.array
      Array of shape (img_h, img_w, img_c) and dtype np.uint8
      the frame to be stored

    Returnsр
    -------
    idx: int
      Index at which the frame is stored. To be used for `store_effect` later.
    """
    self.obs[self.next_idx] = frame

    ret = self.next_idx
    self.next_idx = (self.next_idx + 1) % self.max_size
    self.size_now = min(self.max_size, self.size_now + 1)

    return ret


  def store_effect(self, idx, action, reward, done):
    """Store effects of action taken after obeserving frame stored
    at index idx. The reason `store_frame` and `store_effect` is broken
    up into two functions is so that once can call `encode_recent_obs`
    in between.

    Paramters
    ---------
    idx: int
      Index in buffer of recently observed frame (returned by `store_frame`).
    action: int
      Action that was performed upon observing this frame.
    reward: float
      Reward that was received when the actions was performed.
    done: bool
      True if episode was finished after performing that action.
    """
    self.action[idx] = action
    self.reward[idx] = reward
    self.done[idx]   = done


  def encode_recent_obs(self):
    assert self.size_now > self.state_len
    idx = (self.next_idx - 1) % self.max_size 
    
    if self.state_len == 1:   return self.obs[idx]
    else:                     return self._encode_observation(idx)


  def sample(self, batch_size):
    """
    Sample uniformyl `batch_size` different transitions. Note that the 
    implementation is thread-safe and allows for another thread to be currently
    adding a new transition to the buffer.

    i-th sample transition is as follows: when state `obs[i]`, action 
    `act[i]` was taken. After that reward `rew[i]` was received and subsequent
    state `obs_tp1[i]` was observed. If `done[i]` is True, then the episode was
    finished after taking `act[i]`
    
    Arg:
      batch_size: int. Size of the batch to sample
    Returns:
      Python dictionary with keys 
      "obs": np.array, shape=[batch_size, obs_shape], dtype=obs_dtype, Batch state
        FIX obs_shape - must be times state_len

      "act": np.array, shape=[batch_size, act_shape], dtype=act_dtype. Batch actions
      "rew": np.array, shape=[batch_size, 1], dtype=np.float32. Batch rewards
      "obs_tp1": np.array, shape=[batch_size, obs_shape], dtype=obs_dtype. Batch next state
      "done": np.array, shape=[batch_size, 1], dtype=np.bool. Batch done mask. 
        True if episode has ended, False otherwise
    """
        
    exclude = self._exclude_indices()
    
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
    if self.state_len == 1:
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
    # to the sample that will be overwritten next time. Then:
    # - the idx-1 sample is invalid because the next obs is not the true one
    # - the [idx : idx+state_len-1) samples are have inconsistent history
    # If another thread can be currently incrementing the original value of 
    # self.next_idx, then the actual values that idx can have at this 
    # point are self.next_idx or self.next_idx+1. Then we need to widen the
    # exlude range by 1 at each end. However, the previous sample is no longer
    # guaranteed to be correctly written, so we need to widen the min exclude 
    # range by 1 more
    
    idx     = self.next_idx
    exclude = np.arange(idx-3, idx+self.state_len) % self.max_size
    return exclude


  def _encode_observation(self, idx):
    """Encode the observation for idx by stacking state_len frames preceding
    frames together. This function will always be called when there are more
    than state_len frames in the buffer.

    NOTE: This is used only for image observations
    """
    end_idx   = idx + 1 # make noninclusive
    start_idx = end_idx - self.state_len

    for idx in range(start_idx, end_idx - 1):
      if self.done[idx % self.max_size]:
        start_idx = idx + 1
    missing_context = self.state_len - (end_idx - start_idx)

    # We need to duplicate the start_idx observation
    if missing_context > 0:
      # frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
      frames = [self.obs[start_idx % self.max_size] for _ in range(missing_context)]
      for idx in range(start_idx, end_idx):
        frames.append(self.obs[idx % self.max_size])
      return np.concatenate(frames, 2)
    # We are on the boundary of the buffer
    elif start_idx < 0:
      img_h, img_w = self.obs.shape[1], self.obs.shape[2]
      frames = [self.obs[start_idx:], self.obs[:end_idx]]
      frames = np.concatenate(frames, 0)
      return frames.transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)
    # The standard case
    else:
      # This optimization has potential to saves about 30% compute time \o/
      img_h, img_w = self.obs.shape[1], self.obs.shape[2]
      return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)


  # def encode_observation(self, obs):
  #   """Add obs on top of the most recent `state_len-1` observations.
  #   Args:
  #     obs. np.array, shape=obs_shape, dtype=obs_dtype
  #   Returns:
  #     np.array of the stacked observations
  #     Array of shape (img_h, img_w, img_c * state_len)
  #     and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
  #     encodes frame at time `t - state_len + i`
  #   """
  #   assert self.size_now > 0
  #   return self._encode_observation((self.next_idx - 1) % self.max_size)


  # def store(self, obs, action, reward, done):
  #   """Store effects of action taken after obeserving frame stored
  #   at index idx. The reason `store_frame` and `store_effect` is broken
  #   up into two functions is so that once can call `encode_recent_obs`
  #   in between.

  #   Paramters
  #   ---------
  #   idx: int
  #     Index in buffer of recently observed frame (returned by `store_frame`).
  #   action: int
  #     Action that was performed upon observing this frame.
  #   reward: float
  #     Reward that was received when the actions was performed.
  #   done: bool
  #     True if episode was finished after performing that action.
  #   """
  #   self.obs[self.next_idx]     = action
  #   self.action[self.next_idx]  = action
  #   self.reward[self.next_idx]  = reward
  #   self.done[self.next_idx]    = done

  #   self.next_idx = (self.next_idx + 1) % self.max_size
  #   self.size_now     = min(self.max_size, self.size_now + 1)


