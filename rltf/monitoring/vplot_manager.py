
class TensorPlotConf:
  """Helper configuration object for inserting Tensor values into plots which appear in episode
  video recordings made by rltf.Monitor. Allows for easy communication between the TensorFlow model,
  which is supposed to evaluate the Tensors during a sess.run() call, and the Monitor, which is
  supposed to use the results to create the plots and add them to the recording."""

  def __init__(self):

    self.train_spec   = {}    # dict of tensors that the TF model runs when selecting a train action
    self._train_spec  = None
    self.eval_spec    = {}    # dict of tensors that the TF model runs when selecting an eval action
    self._eval_spec   = None

    self.train_data   = {}    # dict which holds the latest result of sess.run(self.train_spec)
    self.eval_data    = {}    # dict which holds the latest result of sess.run(self.eval_spec)


  def set_train_spec(self, spec):
    """Set the configuration that the model should run during a train step.
    Args:
      spec: dict of str-tf.Tensor pairs
    """
    # NOTE: Do not set self.train_spec. We want it to be deactivated by default
    self._train_spec = spec


  def set_eval_spec(self, spec):
    """Set the configuration that the model should run during an eval step.
    Args:
      spec: dict of str-tf.Tensor pairs
    """
    # NOTE: Do not set self.eval_spec. We want it to be deactivated by default
    self._eval_spec = spec


  def activate_train_plots(self):
    """Activate the tensor configuration so it is run at a train step by the model"""
    self.train_spec = self._train_spec


  def deactivate_train_plots(self):
    """De-activate the tensor configuration so it is NOT run at a train step by the model.
    This increase speed substantially, especially when not all episodes are recorded.
    """
    self.train_spec = {}


  def activate_eval_plots(self):
    """Activate the tensor configuration so it is run at an eval step by the model"""
    self.eval_spec = self._eval_spec


  def deactivate_eval_plots(self):
    """De-activate the tensor configuration so it is NOT run at an eval step by the model.
    This increase speed substantially, especially when not all episodes are recorded.
    """
    self.eval_spec = {}


  @property
  def true_train_spec(self):
    """Return the true train_spec, no matter if activated or not"""
    return self._train_spec


  @property
  def true_eval_spec(self):
    """Return the true train_spec, no matter if activated or not"""
    return self._eval_spec


# Global collection of the TensorPlotConf objects for all models
_COLLECTION = {}


def get_plot_conf(model):
  """Get a reference to the TensorPlotConf for model.
  Args:
    model: str. Name of the model
  Returns:
    The corresponding TensorPlotConf.
  """
  if model not in _COLLECTION:
    _COLLECTION[model] = TensorPlotConf()

  return _COLLECTION[model]
