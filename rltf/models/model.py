import tensorflow as tf


class Model:
  """The base class for operating a Reinforcement Learning deep net in 
  TensorFlow. All networks descend from this class
  """

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
    try: self._train_op       = graph.get_operation_by_name("train_op")
    except KeyError: pass
    try: self._update_target  = graph.get_operation_by_name("update_target")
    except KeyError: pass

    # Get Placeholders
    try: self._obs_t_ph   = graph.get_tensor_by_name("obs_t_ph:0")
    except KeyError: pass
    try: self._act_t_ph   = graph.get_tensor_by_name("act_t_ph:0")
    except KeyError: pass
    try: self._rew_t_ph   = graph.get_tensor_by_name("rew_t_ph:0")
    except KeyError: pass
    try: self._obs_tp1_ph = graph.get_tensor_by_name("obs_tp1_ph:0")
    except KeyError: pass
    try: self._done_ph    = graph.get_tensor_by_name("done_ph:0")
    except KeyError: pass
    try: self._action     = graph.get_tensor_by_name("action:0")
    except KeyError: pass

    self._restore(graph)


  def initialize(self, sess):
    """Run additional initialization for the model when it was created via
    self.build(). Assumes that tf.global_variables_initializer() and
    tf.local_variables_initializer() have already been run
    """
    raise NotImplementedError()


  def control_action(self, sess, state):
    """Compute control action for the model. NOTE that this should NOT include
    any exploration policy, but should only return the action that would be
    performed if the model was being evaluated

    Args:
      sess: tf.Session(). Currently open session
      state: np.array. Observation for the current state
    Returns:
      The calculated action. Type and shape varies based on the specific model
    """
    raise NotImplementedError()


  def _restore(self, graph):
    raise NotImplementedError()


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
    if hasattr(self, "_train_op"): return self._train_op
    else: raise NotImplementedError()

  @property
  def update_target(self):
    """
    Returns:
      `tf.Op` that updates the target network (if one is used).
    """
    if hasattr(self, "_update_target"): return self._update_target
    else: raise NotImplementedError()

  @property
  def obs_t_ph(self):
    """
    Returns:
      `tf.placeholder` for observations at time t from the training batch
    """
    if hasattr(self, "_obs_t_ph"): return self._obs_t_ph
    else: raise NotImplementedError()

  @property
  def act_t_ph(self):
    """
    Returns:
      `tf.placeholder` for actions at time t from the training batch
    """
    if hasattr(self, "_act_t_ph"): return self._act_t_ph
    else: raise NotImplementedError()

  @property
  def rew_t_ph(self):
    """
    Returns:
      `tf.placeholder` for actions at time t from the training batch
    """
    if hasattr(self, "_rew_t_ph"): return self._rew_t_ph
    else: raise NotImplementedError()

  @property
  def obs_tp1_ph(self):
    """
    Returns:
      `tf.placeholder` for observations at time t+1 from the training batch
    """
    if hasattr(self, "_obs_tp1_ph"): return self._obs_tp1_ph
    else: raise NotImplementedError()

  @property
  def done_ph(self):
    """
    Returns:
    `tf.placeholder` to indicate end of episode for examples in the training batch
    """
    if hasattr(self, "_done_ph"): return self._done_ph
    else: raise NotImplementedError()

  # @property
  # def training_ph(self):
  #   return self._training_ph
