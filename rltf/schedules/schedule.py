from abc import ABCMeta, abstractmethod

class Schedule(metaclass=ABCMeta):

  @abstractmethod
  def value(self, t):
    """Value of the schedule at step t"""
    pass

  @abstractmethod
  def __repr__(self):
    """Representation of the schedule for logging purposes"""
    pass
