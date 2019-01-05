import numpy as np

from rltf.memory  import BaseBuffer


class PGBuffer(BaseBuffer):
  """Fixed-size data buffer. Supports both image observations and low-level observations.
  """

  def __init__(self, size, state_shape, obs_dtype, act_shape, act_dtype, obs_len=1):
    """
    Args: See `BaseBuffer.store()`
    """

    super().__init__(size, state_shape, obs_dtype, act_shape, act_dtype, obs_len)

    # Create a buffer for the value function
    vf              = np.empty([self.max_size+1], dtype=np.float32)
    self.vf         = vf[:-1]
    self.next_vf    = vf[1:]
    self.gae_lambda = np.empty([self.max_size], dtype=np.float32)
    self.td_lambda  = np.empty([self.max_size], dtype=np.float32)
    self.logp       = np.empty([self.max_size], dtype=np.float32)

    self.it = None


  #pylint: disable=arguments-differ
  def store(self, obs_t, act_t, rew_tp1, done_tp1, vf_t, logp_t):
    """Store the observed transition defined as: Given `obs_t`, action `act_t` was taken.
    Then reward `reward_tp1` was observed. If after action `act_t` the episode terminated,
    then `done_tp1` will be `True`, otherwise `Fasle`. Note that the observation after taking
    `act_t` should be passed as `obs_t` on the next call to `store()`. NOTE: if `done_tp1 == True`,
    then there is no need to call `store()` on `obs_tp1`: we do NOT need to know it since we never
    use it in computing the backup value
    Args:
      obs_t: See `BaseBuffer.store()`
      act_t: See `BaseBuffer.store()`
      reward_tp1: See `BaseBuffer.store()`
      done_tp1: See `BaseBuffer.store()`
      vf_t: float. Value function estimate for `obs_t`
      logp_t: float. Log probability of action `act_t`
    """

    # Store these before advancing next_idx in BaseBuffer
    self.vf[self.next_idx]    = vf_t
    self.logp[self.next_idx]  = logp_t

    super().store(obs_t, act_t, rew_tp1, done_tp1)


  def __iter__(self):
    self.it = -1
    return self


  def __next__(self):
    if self.it >= self.size_now:
      raise StopIteration
    else:
      self.it += 1
      return self.__getitem__(self.it)


  def __getitem__(self, i):
    # If low-level observations or single frames
    if self.obs_len == 1:
      obs = self.obs[i]
    else:
      obs = self._encode_img_observation(i)

    return obs, self.action[i], self.reward[i], self.done[i], self.vf[i], self.next_vf[i]


  def compute_estimates(self, gamma, lam, next_vf=0):
    """Compute the advantage estimates using the GAE(gamma, lambda) estimator and
    the value function targets using the TD(lambda) estimator
    Args:
      gamma: float. The value of gamma for GAE(gamma, lambda)
      lam: float. The value of lambda for GAE(gamma, lambda) and TD(lambda)
      next_vf: float. The value function estimate for the observation encountered after the
        last step. Must be 0 if the episode was done
    """

    # Assert that the buffer is exactly filled
    assert self.next_idx == 0

    self.next_vf[-1] = next_vf
    gae_t = 0

    # Compute GAE(gamma, lambda)
    # pylint: disable=redefined-argument-from-local
    for t, (_, _, rew, done, vf, next_vf) in zip(reversed(range(self.size_now)), reversed(self)):
      delta_t = rew + (1 - done) * gamma * next_vf  - vf
      gae_t   = delta_t + (1 - done) * gamma * lam * gae_t
      self.gae_lambda[t] = gae_t

    # Compute TD(lambda)
    self.td_lambda = self.gae_lambda + self.vf


  def get_data(self):
    """Return all data"""
    return self._batch_samples(np.arange(0, self.size_now))


  def iterate(self, batch_size, shuffle=True):

    size = (self.size_now // batch_size) * batch_size

    inds = np.arange(0, self.size_now)
    if shuffle:
      # inds = self.prng.shuffle(inds)[:size]
      self.prng.shuffle(inds)
    inds = inds[:size]

    for lo in range(0, size, batch_size):
      hi = lo + batch_size
      yield self._batch_samples(inds[lo:hi])


  def _batch_samples(self, inds):
    """Takes the samples from the buffer stacks them into a batch
    Args:
      inds: np.array or list. Indices for transitions to be sampled from the buffer
    Returns:
      See self.sample()
    """
    if self.obs_len == 1:
      obs_batch     = self.obs[inds]
    else:
      obs_batch     = np.concatenate([self._encode_img_observation(idx)[None] for idx in inds], 0)

    act_batch   = self.action[inds]
    gae_batch   = self.gae_lambda[inds]
    td_batch    = self.td_lambda[inds]
    logp_batch  = self.logp[inds]
    vf_batch    = self.vf[inds]

    return dict(obs=obs_batch, act=act_batch, adv=gae_batch, ret=td_batch, logp=logp_batch, vf=vf_batch)
