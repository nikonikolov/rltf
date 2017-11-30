import numpy as np

from rltf.exploration.exploration import ExplorationNoise


class NoNoise(ExplorationNoise):
  """Returns 0 as noise"""

  def __init__(self, mu):
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
    
    self.mu = mu
    self.sigma = sigma

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
  """
  
  def __init__(self, mu, sigma, theta=0.15, dt=1e-2):
    """
    Args:
      mu: np.array or int. Noise mean
      sigma: np.array or int. Wiener noise scale constant. Should have the same shape as mu
      theta: float. Mean attraction constant
      dt: float. Time constant. Can be interpreted as the time difference 
        between two environent actions
    """
    super().__init__()

    try:
      self.shape = mu.shape
    except AttributeError:
      mu = float(mu)
      try:
        self.shape = sigma.shape
      except AttributeError:
        sigma = float(sigma)
        self.shape = None

    self.mu     = mu
    self.sigma  = sigma
    self.theta  = theta
    self.dt     = dt
    self.reset()

  def sample(self):
    # self.x += self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.shape)
    mean  = self.theta * (self.mu - self.x) * self.dt
    std   = self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.shape)
    self.x += mean + std
    return self.x

  def reset(self):
    self.x = np.zeros_like(self.mu)

  def __repr__(self):
    return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={}, theta={})'.format(
      self.mu, self.sigma, self.theta, self.dt)
