import tensorflow as tf

from rltf.schedules import ConstSchedule
from rltf.schedules import Schedule


class OptimizerConf:
  """Config for an optimizer"""

  def __init__(self, opt_type, learn_rate, **kwargs):
    """
    Args:
      opt_type: tf.train.Optimizer subclass. Constructor for the optimizer, e,g, tf.train.AdamOptimizer
      learn_rate: float or rltf.schedules.Schedule. Schedule for the learning rate
      kwargs: dict. All additional keyword arguments will be passed on directly
        to the optimizer constructor
    """
    if learn_rate is None or isinstance(learn_rate, float):
      lr_schedule = ConstSchedule(learn_rate)
    elif isinstance(learn_rate, Schedule):
      lr_schedule = learn_rate
    else:
      raise TypeError("Incorrect learn_rate type {}".format(type(learn_rate)))

    self.opt_type     = opt_type
    self.lr_schedule  = lr_schedule
    self.kwargs       = kwargs
    self.lr_ph        = None
    self.built        = False
    self.opt          = None


  def build(self, lr_tb_name=None, lr_ph_name=None):
    """Construct the optimizer with all the specs and a learning rate placeholder
    Args:
      lr_tb_name: str or None. Name for a tensorboard scalar summary to attach to the learning rate.
        If None, no summary is attached.
      lr_ph_name: str. Optional name for the placeholder Tensor
    Returns:
      The built optimizer. Instance of tf.train.Optimizer
    """
    if self.built:
      return self.opt
    self.built = True

    self.lr_ph = tf.placeholder(tf.float32, shape=(), name=lr_ph_name)

    if lr_tb_name is not None:
      tf.summary.scalar(lr_tb_name, self.lr_ph)

    self.opt = self.opt_type(learning_rate=self.lr_ph, **self.kwargs)

    return self.opt


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
