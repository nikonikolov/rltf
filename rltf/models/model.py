import collections
import logging
import tensorflow as tf


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

    # plot_train: UserDict of `tf.Tensor`(or `np.array`) objects that have to be run in the session
    # plot_data: UserDict of `np.array`s that contain the actual data to be plotted
    # For performance `plot_train.data` should be modified from the outside to determine when to run
    # the tensors and transfer data between CPU and GPU. `plot_data.data` should be set to the result
    # of `sess.run(plot_train)` after every step, no matter the value of `plot_train.data`.
    # `plot_data` will be accessed from the outside to fetch the data when needed
    self.plot_train = collections.UserDict()
    self.plot_eval  = collections.UserDict()
    self.plot_data  = collections.UserDict()

    # Regex that matches variables that should not be trained. Used when variable values are
    # restored and reused from an already trained model
    self.notrain_re = None

    # List of all model variables
    self._variables = None


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


  def action_train(self, sess, state):
    """Compute the training action from the model. NOTE that this should NOT include
    any exploration policy, but should only return the action that would be
    performed if the model was being evaluated
    Args:
      sess: tf.Session(). Currently open session
      state: np.array. Observation for the current state
    Returns:
      The calculated action. Type and shape varies based on the specific model
    """
    raise NotImplementedError()


  def action_eval(self, sess, state):
    """Compute the action that should be taken in evaluation mode
    Args:
      sess: tf.Session(). Currently open session
      state: np.array. Observation for the current state
    Returns:
      The calculated action. Type and shape varies based on the specific model
    """
    raise NotImplementedError()


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


  def clear_plot_tensors(self):
    """Clear dicts with plot tensors in order to avoid running them every time"""
    self.plot_train.data = dict()
    self.plot_eval.data  = dict()
    self.plot_data.data  = dict()


  def _update_plot_data(self, data):
    self.plot_data.data = data


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
    if self._variables is not None:
      return self._variables
    else:
      raise NotImplementedError()
