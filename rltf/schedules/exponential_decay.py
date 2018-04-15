import math

from rltf.schedules.schedule  import Schedule
from rltf.schedules.utils     import linear_interpolation


class ExponentialDecay(Schedule):

  def __init__(self, init, final, decay_rate):
    """Exponential decay schedule starting from `init`. At timestep `t` the output is
    `init * decay_rate^t`. The minimum/maximum possible output is `final`

    Args:
      init: float. Initial output value
      final: float. Final output value
      decay: float. Decay factor
    """
    assert decay_rate > 0

    self.decay_rate = float(decay_rate)
    self.final      = float(final)
    self.init       = float(init)

    self.t          = 1
    self.v          = self.init
    self.max_t      = int(math.log(self.final/self.init, self.decay_rate)) + 1


  def value(self, t):
    """See Schedule.value"""
    if t > self.max_t:
      return self.final

    # Compute the exponential difference
    diff = t - self.t
    v = self.v * math.pow(self.decay_rate, diff)
    self.v = v
    self.t = t
    return v


  def __repr__(self):
    string = self.__class__.__name__
    string += "(initial={}, final={}, decay_rate={})".format(self.init, self.final, self.decay_rate)
    return string
