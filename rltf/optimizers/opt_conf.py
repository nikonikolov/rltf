
class OptimizerConf:
  """Config for an optimizer"""

  def __init__(self, opt_type, lr_schedule, **kwargs):
    """
    Args:
      opt_type: tf.train.Optimizer. Constructor for the optimizer, e,g, tf.train.AdamOptimizer
      lr_schedule: rltf.schedules.Schedule. Schedule for the learning rate
      kwargs: dict. All additional keyword arguments will be passed on directly
        to the optimizer constructor
    """

    self.opt_type     = opt_type
    self.lr_schedule  = lr_schedule
    self.kwargs       = kwargs
    self.lr_ph        = None


  def build(self):
    """Construct the optimizer with all the specs and return it"""
    return self.opt_type(self.lr_ph, **self.kwargs)


  def lr_value(self, t):
    """
    Args:
      t: current timestep
    Returns:
      The value of the learning rate schedule for timestep t
    """
    return self.lr_schedule.value(t)


  def __repr__(self):
    string = self.opt_type.__name__ + '(learn_rate={}'.format(self.lr_schedule)
    for a, v in self.kwargs.items():
      string += ", {}={}".format(a, v)
    string += ")"
    return string
