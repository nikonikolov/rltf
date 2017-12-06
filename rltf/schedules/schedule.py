
class Schedule:

  def value(self, t):
    """Value of the schedule at step t"""
    raise NotImplementedError()

  def __repr__(self):
    """Representation of the schedule for logging purposes"""
    raise NotImplementedError()
