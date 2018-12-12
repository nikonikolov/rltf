import logging
import tensorflow as tf

from rltf.monitoring import vplot_manager

logger = logging.getLogger(__name__)


class Model:
  """The base class for operating a Reinforcement Learning deep net in TensorFlow.
  All network estimators descend from this class
  """

  def __init__(self):

    # Properties that should be set by the subclass
    self.obs_dtype  = None
    self.obs_shape  = None
    self.act_dtype  = None
    self.act_shape  = None

    # Get a TensorPlotConf object that manages plotting tenors in episode video recordings
    self.plot_conf  = vplot_manager.get_plot_conf(self.name)

    self.train_dict = None  # Dict of all tensors to run when fetching a train action
    self.eval_dict  = None  # Dict of all tensors to run when fetching an eval action
    self.ops_dict   = {}    # Dict of all important model tensors. Used for general access (by agent)

    # Regex that matches variables that should not be trained. Used when variable values are
    # restored and reused from an already trained model
    self.notrain_re = None

    # List of all model variables
    self._vars      = None


  def build(self):
    raise NotImplementedError()


  def initialize(self, sess):
    """Run additional initialization for the model when it was created via
    self.build(). Assumes that tf.global_variables_initializer() and
    tf.local_variables_initializer() have already been run
    """
    raise NotImplementedError()


  def reset(self, sess):
    """This method is called by the agent at the end of every episode. Allows for
    internal changes in the model that stay the same for the duration of the whole episode
    """
    raise NotImplementedError()


  def action_train_ops(self, sess, state, run_dict=None):
    """Compute the training action from the model and any additional tensors.
    Args:
      sess: tf.Session(). Currently open session
      state: np.array. Observation for the current state
      run_dict: dict of str-tf.Tensor pairs. Contains any additional tensors to run
    Returns:
      dict of str-np.array pairs. Contains the action, additional model tensors and run_dict
    """
    raise NotImplementedError()


  def action_eval_ops(self, sess, state, run_dict=None):
    """Compute the action that should be taken in evaluation mode and any additional tensors.
    Args:
      sess: tf.Session(). Currently open session
      state: np.array. Observation for the current state
      run_dict: dict of str-tf.Tensor pairs. Contains any additional tensors to run
    Returns:
      dict of str-np.array pairs. Contains the action, additional model tensors and run_dict
    """
    raise NotImplementedError()


  def _action_train_ops(self, sess, run_dict, feed_dict):
    if run_dict is None:
      run_dict = self.train_dict
    else:
      run_dict = {**run_dict, **self.train_dict}

    # Run the results and update any data that needs to be plotted
    data, self.plot_conf.train_data = sess.run([run_dict, self.plot_conf.train_spec], feed_dict=feed_dict)
    return data


  def _action_eval_ops(self, sess, run_dict, feed_dict):
    if run_dict is None:
      run_dict = self.eval_dict
    else:
      run_dict = {**run_dict, **self.eval_dict}

    # Run the results and update any data that needs to be plotted
    data, self.plot_conf.eval_data = sess.run([run_dict, self.plot_conf.eval_spec], feed_dict=feed_dict)
    return data


  def exclude_train_vars(self, regex):
    """Set a regex to match and exclude model variables which should not be trained
    Args:
      regex: A compiled Regular Expression Object from the `re` module. Must support `regex.search()`
    """
    self.notrain_re = regex


  def _trainable_variables(self, scope):
    """Get the trainable variables in the given scope and remove any which match `self.notrain_re`
    Args:
      scope: str. TensorFlow variable scope
    Returns:
      list of `tf.Variable`s that should be trained
    """
    train_vars = tf.trainable_variables(scope=scope)
    # Exlude variables that are explicitly exluded from training
    if self.notrain_re is not None:
      exlcude = [v for v in train_vars if self.notrain_re.search(v.name)]
      if len(exlcude) > 0:
        logger.info("Excluding model variables in '%s' scope from training:", scope)
        for v in exlcude:
          logger.info(v.name)
        train_vars = [v for v in train_vars if v not in exlcude]
      else:
        logger.info("No variables in scope '%s' will be excluded from training:", scope)
    return train_vars


  @property
  def name(self):
    """
    Returns:
      name of the model class
    """
    return self.__class__.__name__


  @property
  def variables(self):
    """
    Returns:
      `list` of `tf.Variable`s which contains all variables used by the model. If there is a target
      network, its variables must be included. Optimizer related variables must be excluded
    """
    if self._vars is not None:
      return self._vars
    else:
      raise NotImplementedError()
