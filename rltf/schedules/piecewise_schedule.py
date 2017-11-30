from rltf.schedules.schedule  import Schedule
from rltf.schedules.utils     import linear_interpolation


class PiecewiseSchedule(Schedule):
  def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
    """Piecewise schedule.

    Args:
      endpoints: list of pairs [(int, int)]. Every pair `(time, value)` means
       that schedule should output `value` when `t==time`. All time values 
       must be sorted in an increasing order. For times in between endpoints, 
       interpolation is used to return a value
      interpolation: lambda float, float, float: float
        a function that takes value to the left and to the right of t according
        to the `endpoints`. The last argument alpha is the fraction of distance 
        from left endpoint to right endpoint that t has covered. 
        See linear_interpolation for example.
      outside_value: float
        if the value is requested outside of all the intervals sepecified in
        `endpoints` this value is returned.
    """
    idxes = [e[0] for e in endpoints]
    assert idxes == sorted(idxes)
    self._interpolation = interpolation
    self._outside_value = outside_value
    self._endpoints     = endpoints

  def value(self, t):
    """See Schedule.value

    Raises:
      AssertionError if outside_value is None and outside value is requested.
    """
    for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
      if l_t <= t and t < r_t:
        alpha = float(t - l_t) / (r_t - l_t)
        return self._interpolation(l, r, alpha)

    # t does not belong to any of the pieces, so doom.
    assert self._outside_value is not None
    return self._outside_value


  def __repr__(self):
    string = self.__class__.__name__ + "("
    for step, val in self._endpoints:
      string += (" (%d, %f);" % (step, val))
    string += " )"
    return string
