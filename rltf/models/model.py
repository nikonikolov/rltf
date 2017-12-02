import tensorflow as tf


class Model:
  """The base class for operating a Reinforcement Learning deep net in 
  TensorFlow. All networks descend from this class
  """

  def build(self, optimizer):
    raise NotImplementedError()

  def restore(self, ckpt_path):
    """Restore the TF graph (without the session) saved on disk in model_dir.
    The object resulting state must be equivalent to a call to self.build_graph()
    
    Args:
      ckpt_path: str. Path for the checkpoint
    Returns:
      tf.Saver that is used to restore the session
    """
    raise NotImplementedError()

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

  # QUESTION: Should this be a variable or a placeholder? Maybe a variable???
  @property
  def training_ph(self):
    return self._training_ph

  def _restore(self, ckpt_path):
    saver = tf.train.import_meta_graph(ckpt_path + '.meta')
    graph = tf.get_default_graph()
    
    # Get Ops
    try: self._train_op       = graph.get_operation_by_name("train_op")
    except KeyError: pass
    try: self._update_target  = graph.get_operation_by_name("update_target")
    except KeyError: pass
    
    # Get Placeholders
    try: self._obs_t_ph     = graph.get_tensor_by_name("obs_t_ph:0")
    except KeyError: pass
    try: self._act_t_ph     = graph.get_tensor_by_name("act_t_ph:0")
    except KeyError: pass
    try: self._rew_t_ph     = graph.get_tensor_by_name("rew_t_ph:0")
    except KeyError: pass
    try: self._obs_tp1_ph   = graph.get_tensor_by_name("obs_tp1_ph:0")
    except KeyError: pass
    try: self._done_mask_ph = graph.get_tensor_by_name("done_ph:0")
    except KeyError: pass
    try: self._action       = graph.get_tensor_by_name("action:0")
    except KeyError: pass

    return graph, saver


  # @property
  # def agent_vars(self):
  #   """
  #   Returns:
  #   `list` of the variables used for estimator network
  #   """
  #   if hasattr(self, "_agent_vars"):
  #     return self._agent_vars
  #   else:
  #     raise NotImplementedError()

  # @property
  # def _target_vars(self):
  #   """
  #   Returns:
  #   `list` of the variables used for the target network (if such exists)
  #   """
  #   if hasattr(self, "_target_vars"):
  #     return self._target_vars
  #   else:
  #     raise NotImplementedError()

  # @property
  # def vars(self):
  #   raise NotImplementedError()

  # @property
  # def trainable_vars(self):
  #   raise NotImplementedError()

  # @property
  # def perturbable_vars(self):
  #   raise NotImplementedError()
