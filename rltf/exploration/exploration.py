from abc import ABCMeta, abstractmethod

from rltf.utils import seeding


class ExplorationNoise(metaclass=ABCMeta):

  def __init__(self):
    self.prng = seeding.get_prng()

  @abstractmethod
  def sample(self, t):
    """Get a sample from the noise process for the given time step
    Args:
      t: Current time step
    Returns:
      float: the sampled noise value
    """
    pass

  @abstractmethod
  def reset(self):
    """Reset the noise process"""
    pass
