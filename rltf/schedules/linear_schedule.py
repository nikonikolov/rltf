from rltf.schedules.schedule  import Schedule
from rltf.schedules.utils     import linear_interpolation


class LinearSchedule(Schedule):

  def __init__(self, timesteps, final_p, initial_p=1.0):
    """Linear interpolation between initial_p and final_p over timesteps. 
    After this many timesteps pass final_p is returned.
    
    Args:
      timesteps: int. Number of timesteps for which to linearly anneal initial_p to final_p
      initial_p: float. Initial output value
      final_p: float. Final output value
    """
    self.timesteps = timesteps
    self.final_p   = final_p
    self.initial_p = initial_p


  def value(self, t):
    """See Schedule.value"""
    fraction  = min(float(t) / self.timesteps, 1.0)
    return self.initial_p + fraction * (self.final_p - self.initial_p)


  def __repr__(self):
    string = self.__class__.__name__
    string += "(initial={}, final={}, timesteps={})".format(self.initial_p, self.final_p, self.timesteps)
    return string
