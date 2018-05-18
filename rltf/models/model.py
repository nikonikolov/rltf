import collections
import logging
import tensorflow as tf


logger = logging.getLogger(__name__)


class Model:
  """The base class for operating a Reinforcement Learning deep net in
  TensorFlow. All networks descend from this class
  """

  def __init__(self):
    # Input TF placeholders that must be set
    self._obs_t_ph      = None
    self._act_t_ph      = None
    self._rew_t_ph      = None
    self._obs_tp1_ph    = None
    self._done_ph       = None

    # Properties that should be set by the subclass
    self.obs_dtype      = None
    self.obs_shape      = None
    self.act_dtype      = None
    self.act_shape      = None

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

    # TF Ops that should be set
    self._train_op      = None
    self._update_target = None  # Optional
    self._variables     = None


  def build(self):
    raise NotImplementedError()


  def _build(self):
    self._obs_t_ph    = tf.placeholder(self.obs_dtype,  [None] + self.obs_shape, name="obs_t_ph")
    self._act_t_ph    = tf.placeholder(self.act_dtype,  [None] + self.act_shape, name="act_t_ph")
    self._rew_t_ph    = tf.placeholder(tf.float32,      [None],                  name="rew_t_ph")
    self._obs_tp1_ph  = tf.placeholder(self.obs_dtype,  [None] + self.obs_shape, name="obs_tp1_ph")
    self._done_ph     = tf.placeholder(tf.bool,         [None],                  name="done_ph")


  def restore(self, graph):
    """Restore the Variables, placeholders and Ops needed by the class so that
    it can operate in exactly the same way as if `self.build()` was called
    Args:
      graph: tf.Graph. Graph, restored from a checkpoint
    """

    # Get Ops
    try:
      self._train_op       = graph.get_operation_by_name("train_op")
    except KeyError:
      pass
    try:
      self._update_target  = graph.get_operation_by_name("update_target")
    except KeyError:
      pass

    # Get Placeholders
    try:
      self._obs_t_ph   = graph.get_tensor_by_name("obs_t_ph:0")
    except KeyError:
      pass

    try:
      self._act_t_ph   = graph.get_tensor_by_name("act_t_ph:0")
    except KeyError:
      pass

    try:
      self._rew_t_ph   = graph.get_tensor_by_name("rew_t_ph:0")
    except KeyError:
      pass

    try:
      self._obs_tp1_ph = graph.get_tensor_by_name("obs_tp1_ph:0")
    except KeyError:
      pass

    try:
      self._done_ph    = graph.get_tensor_by_name("done_ph:0")
    except KeyError:
      pass

    self._restore(graph)


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


  def _restore(self, graph):
    raise NotImplementedError()


  def exlcude_train_vars(self, regex):
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
  def train_op(self):
    """
    Returns:
      `tf.Op` that trains the network. Requires that `self.obs_t_ph`,
      `self.act_t_ph`, `self.obs_tp1_ph`, `self.done_ph` placeholders
      are set via feed_dict. Might require other placeholders as well.
    """
    if self._train_op is not None:
      return self._train_op
    else:
      raise NotImplementedError()

  @property
  def update_target(self):
    """
    Returns:
      `tf.Op` that updates the target network (if one is used).
    """
    if self._update_target is not None:
      return self._update_target
    else:
      raise NotImplementedError()

  @property
  def obs_t_ph(self):
    """
    Returns:
      `tf.placeholder` for observations at time t from the training batch
    """
    if self._obs_t_ph is not None:
      return self._obs_t_ph
    else:
      raise NotImplementedError()

  @property
  def act_t_ph(self):
    """
    Returns:
      `tf.placeholder` for actions at time t from the training batch
    """
    if self._act_t_ph is not None:
      return self._act_t_ph
    else:
      raise NotImplementedError()

  @property
  def rew_t_ph(self):
    """
    Returns:
      `tf.placeholder` for actions at time t from the training batch
    """
    if self._rew_t_ph is not None:
      return self._rew_t_ph
    else:
      raise NotImplementedError()

  @property
  def obs_tp1_ph(self):
    """
    Returns:
      `tf.placeholder` for observations at time t+1 from the training batch
    """
    if self._obs_tp1_ph is not None:
      return self._obs_tp1_ph
    else:
      raise NotImplementedError()

  @property
  def done_ph(self):
    """
    Returns:
      `tf.placeholder` to indicate end of episode for examples in the training batch
    """
    if self._done_ph is not None:
      return self._done_ph
    else:
      raise NotImplementedError()

  # @property
  # def training_ph(self):
  #   return self._training_ph

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
