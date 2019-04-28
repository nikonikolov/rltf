from abc import ABCMeta, abstractmethod

import tensorflow as tf

from rltf.models    import Model
from rltf.tf_utils  import tf_utils


class BaseQlearn(Model, metaclass=ABCMeta):
  """Abstract Q-learning class"""

  def __init__(self):

    super().__init__()

    # Input TF placeholders that must be set
    self.obs_t_ph   = None
    self.act_t_ph   = None
    self.rew_t_ph   = None
    self.obs_tp1_ph = None
    self.done_ph    = None

    # TF Ops that should be set
    self.train_op       = None
    self.update_target  = None  # Optional


  def _build_ph(self):
    """Build the input placehodlers"""
    self.obs_t_ph   = tf.placeholder(self.obs_dtype,  [None] + self.obs_shape, name="obs_t_ph")
    self.act_t_ph   = tf.placeholder(self.act_dtype,  [None] + self.act_shape, name="act_t_ph")
    self.rew_t_ph   = tf.placeholder(tf.float32,      [None],                  name="rew_t_ph")
    self.obs_tp1_ph = tf.placeholder(self.obs_dtype,  [None] + self.obs_shape, name="obs_tp1_ph")
    self.done_ph    = tf.placeholder(tf.bool,         [None],                  name="done_ph")



class BaseDQN(BaseQlearn):
  """Abstract DQN class"""

  def __init__(self, obs_shape, n_actions, opt_conf, gamma):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
    """

    assert len(obs_shape) == 3 or len(obs_shape) == 1

    super().__init__()

    self.gamma      = gamma
    self.opt_conf   = opt_conf

    self.obs_dtype  = tf.uint8 if len(obs_shape) == 3 else tf.float32
    self.obs_shape  = obs_shape
    self.act_dtype  = tf.uint8
    self.act_shape  = []
    self.n_actions  = n_actions

    # Custom TF Tensors and Ops
    self.obs_t      = None
    self.obs_tp1    = None


  def build(self):

    # Build the input placeholders
    self._build_ph()

    # Preprocess the observation
    self.obs_t    = tf_utils.preprocess_input(self.obs_t_ph)
    self.obs_tp1  = tf_utils.preprocess_input(self.obs_tp1_ph)

    # Construct the Q-network and the target network
    agent_out     = self._nn_model(self.obs_t,   scope="agent_net")
    target_out    = self._nn_model(self.obs_tp1, scope="target_net")

    # Compute the estimated Q-function and its backup value
    estimate      = self._compute_estimate(agent_out)
    target        = self._compute_target(target_out)

    # Compute the loss
    loss          = self._compute_loss(estimate, target, name="train/loss")

    train_vars    = self._trainable_variables(scope="agent_net")
    agent_vars    = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_net")
    target_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")

    # Build the optimizer and the train op
    optimizer     = self.opt_conf.build(lr_tb_name="train/learn_rate")
    train_op      = self._build_train_op(optimizer, loss, train_vars, name="train_op")

    # Create the Op to update the target
    update_target = tf_utils.assign_vars(target_vars, agent_vars, name="update_target")

    # Compute the train and eval actions
    self.train_dict = self._act_train(agent_out, name="a_train")
    self.eval_dict  = self._act_eval(agent_out,  name="a_eval")

    self.train_op       = train_op
    self.update_target  = update_target
    self._vars          = agent_vars + target_vars


  def _nn_model(self, x, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      if len(self.obs_shape) == 3:
        return self._conv_nn(x)
      else:
        return self._dense_nn(x)


  @abstractmethod
  def _conv_nn(self, x):
    pass


  def _dense_nn(self, x):
    raise NotImplementedError()


  @abstractmethod
  def _act_train(self, agent_net, name):
    pass


  @abstractmethod
  def _act_eval(self, agent_net, name):
    pass


  @abstractmethod
  def _compute_estimate(self, agent_net):
    pass


  def _compute_target(self, target_net):
    target = self._select_target(target_net)
    target = tf.identity(target, name="target")
    backup = self._compute_backup(target)
    backup = tf.identity(backup, name="backup")
    backup = tf.stop_gradient(backup)
    return backup


  @abstractmethod
  def _select_target(self, target_net):
    pass


  @abstractmethod
  def _compute_backup(self, target):
    pass


  @abstractmethod
  def _compute_loss(self, estimate, target, name):
    pass


  def _build_train_op(self, optimizer, loss, agent_vars, name):
    grads     = self._compute_gradients(optimizer, loss, agent_vars)
    train_op  = optimizer.apply_gradients(grads, name=name)
    return train_op


  def _compute_gradients(self, optimizer, loss, agent_vars, gate_grads=True):
    grads = optimizer.compute_gradients(loss, var_list=agent_vars)
    if gate_grads:
      grads = tf_utils.gate_gradients(grads)
    return grads


  def initialize(self, sess):
    """Initialize the model. See Model.initialize()"""
    sess.run(self.update_target)


  def reset(self, sess):
    pass


  def action_train_ops(self, sess, state, run_dict=None):
    return super()._action_train_ops(sess, run_dict, feed_dict={self.obs_t_ph: state[None,:]})


  def action_eval_ops(self, sess, state, run_dict=None):
    return super()._action_eval_ops(sess, run_dict, feed_dict={self.obs_t_ph: state[None,:]})
