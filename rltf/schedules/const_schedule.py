from rltf.schedules.schedule import Schedule

class ConstSchedule(Schedule):

  def __init__(self, value):
    """Value remains constant over time.
    Args:
      value: float. The constant value of the schedule
    """
    self._v = value

  def value(self, t):
    """See Schedule.value"""
    return self._v

  def __repr__(self):
    """See Schedule.__str__"""
    string = self.__class__.__name__ + "({})".format(self._v)
    return string
