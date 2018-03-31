import numpy as np

from rltf.exploration.exploration import ExplorationNoise


class NoNoise(ExplorationNoise):
  """Returns 0 as noise"""

  def __init__(self):
    super().__init__()

  def sample(self):
    return 0.0

  def reset(self):
    pass

  def __repr__(self):
    return 'NoNoise()'


class GaussianNoise(ExplorationNoise):
  """Produces Gaussian Noise"""

  def __init__(self, mu, sigma):
    """
    Args:
      mu: float or np.array. Mean of the Gaussian
      sigma: float or np.array. Standard deviation of the Gaussian
    """
    super().__init__()

    self.mu     = mu
    self.sigma  = sigma

  def sample(self):
    return np.random.normal(self.mu, self.sigma)

  def reset(self):
    pass

  def __repr__(self):
    return 'GaussianNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class OrnsteinUhlenbeckNoise(ExplorationNoise):
  """Simulates Ornstein-Uhlenbeck Random Process

  Sources:
    - https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    - https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    - https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    - https://github.com/rll/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
  """

  def __init__(self, mu, sigma, theta=0.15, dt=1e-2):
    """
    Args:
      mu: np.array or scalar. Noise mean
      sigma: np.array or scalar. Wiener noise scale constant. Should have the same shape as mu
      theta: float. Mean attraction constant
      dt: float. Time constant. Can be interpreted as the time difference
        between two environent actions
    """
    super().__init__()

    assert isinstance(sigma, np.ndarray)
    assert isinstance(mu,    np.ndarray)
    assert mu.shape == sigma.shape

    self.mu     = mu
    self.sigma  = sigma
    self.theta  = theta
    self.dt     = dt
    self.x      = None
    self.reset()

  def sample(self):
    x = self.x + self.theta * (self.mu - self.x) * self.dt + \
        self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.sigma.shape)
    self.x = x
    return x

  def reset(self):
    self.x = self.mu
    # self.x = np.zeros_like(self.mu, dtype=np.float32)

  def __repr__(self):
    return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={}, theta={}, dt={})'.format(
      self.mu, self.sigma, self.theta, self.dt)
