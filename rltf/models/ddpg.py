import logging
import tensorflow as tf

from rltf.models import BaseQlearn
from rltf.models import tf_utils


logger = logging.getLogger(__name__)


def init_hidden_uniform():
  return tf.variance_scaling_initializer(scale=1.0/3.0, mode="fan_in", distribution="uniform")

def init_output_uniform():
  return tf.random_uniform_initializer(-3e-3, 3e-3)

def init_output_uniform_conv():
  return tf.random_uniform_initializer(-3e-4, 3e-4)


class DDPG(BaseQlearn):

  def __init__(self, obs_shape, act_shape, actor_opt_conf, critic_opt_conf, critic_reg,
               tau, gamma, batch_norm, obs_norm, critic_huber_loss=False):
    """
    Args:
      obs_shape: list. Shape of a single state input
      act_shape: list of 1 element. Shape of a single action
      actor_opt_conf: rltf.optimizers.OptimizerConf. Configuration for the actor optimizer
      critic_opt_conf: rltf.optimizers.OptimizerConf. Configuration for the critic optimizer
      critic_reg: float. Regularization parameter for the weights of the critic
      tau: float. Rate of update for the target network
      gamma: float. Discount factor
      batch_norm: bool. Whether to add batch normalization layers
      obs_norm: bool. Whether to normalize input observations
      critic_huber_loss: bool. Whether to use Huber loss or not
    """

    super().__init__()

    self.tau          = tau
    self.huber_loss   = critic_huber_loss
    self.gamma        = gamma
    self.critic_reg   = critic_reg
    self.batch_norm   = batch_norm
    self.obs_norm     = obs_norm
    self._actor       = None
    self._critic      = None

    self.actor_opt_conf   = actor_opt_conf
    self.critic_opt_conf  = critic_opt_conf

    self.obs_shape = obs_shape
    self.obs_dtype = tf.uint8 if len(obs_shape) == 3 else tf.float32
    self.n_actions = act_shape[0]
    self.act_shape = act_shape
    self.act_dtype = tf.float32

    # Initializers
    self.hidden_init = init_hidden_uniform
    self.output_init = init_output_uniform

    # Custom TF Tensors and Ops
    self._init_op   = None
    self._training  = None
    self._action    = None

    # Set the correct actor and critic networks
    if len(self.obs_shape) == 3 and self.obs_dtype == tf.uint8:
      self._actor   = self._actor_conv_net
      self._critic  = self._critic_conv_net
    elif len(self.obs_shape) == 1 and self.obs_dtype == tf.float32 or self.obs_dtype == tf.float64:
      self._actor   = self._actor_net
      self._critic  = self._critic_net
    else:
      raise ValueError("Invalid observation shape and type")


  def build(self):

    # Build the input placeholders
    self._build_ph()

    # Preprocess the observations
    obs_t   = tf_utils.preprocess_input(self.obs_t_ph,   norm=self.obs_norm, training=self._training)
    obs_tp1 = tf_utils.preprocess_input(self.obs_tp1_ph, norm=self.obs_norm, training=self._training)

    actor           = self._actor(obs_t,                  scope="agent_net/actor")
    actor_critic_q  = self._critic(obs_t, actor,          scope="agent_net/critic")
    act_t_q         = self._critic(obs_t, self.act_t_ph,  scope="agent_net/critic")

    target_act      = self._actor(obs_tp1,                scope="target_net/actor")
    target_q        = self._critic(obs_tp1, target_act,   scope="target_net/critic")

    agent_vars      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_net")
    target_vars     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
    actor_vars      = self._trainable_variables(scope="agent_net/actor")
    critic_vars     = self._trainable_variables(scope="agent_net/critic")

    target_q        = self._compute_target(target_q)

    actor_loss      = self._get_actor_loss(actor_critic_q)
    critic_loss     = self._get_critic_loss(target_q, act_t_q)

    # Build the optimizers
    actor_opt       = self.actor_opt_conf.build(lr_tb_name="train/actor_learn_rate")
    critic_opt      = self.critic_opt_conf.build(lr_tb_name="train/critic_learn_rate")

    # Create train Op
    self._train_op  = self._build_train_op(actor_opt, critic_opt, actor_loss, critic_loss, actor_vars, critic_vars)

    # Create the Op that updates the target
    logger.debug("Creating target net update Op")
    self._update_target = tf_utils.assign_vars(target_vars, agent_vars, self.tau, "update_target")

    # Remember the action tensor. name is needed when restoring the graph
    self.train_dict = dict(action=tf.identity(actor, name="action"))
    self.eval_dict  = self.train_dict

    # Initialization Op
    logger.debug("Creating initialization Op")
    self._init_op   = tf_utils.assign_vars(target_vars, agent_vars, name="init_op")

    # Rememeber the model variables
    self._vars      = agent_vars + target_vars

    self._add_summaries(actor_loss, critic_loss, act_t_q, target_q)


  def _build_ph(self):
    super()._build_ph()

    # Placehodler for the running mode - training or evaluation
    self._training  = tf.placeholder_with_default(True, (), name="training")


  def _add_summaries(self, actor_loss, critic_loss, act_t_q, target_q):
    # Summaries
    tf.summary.scalar("train/actor_loss",   actor_loss)
    tf.summary.scalar("train/critic_loss",  critic_loss)

    tf.summary.scalar("train/actor_critic_q", -actor_loss)
    tf.summary.scalar("train/act_t_q",        tf.reduce_mean(act_t_q))
    tf.summary.scalar("train/target_q",       tf.reduce_mean(target_q))


  def _compute_target(self, target_q):
    done_mask = tf.cast(tf.logical_not(self.done_ph), tf.float32)
    done_mask = tf.expand_dims(done_mask, axis=-1)
    reward    = tf.expand_dims(self.rew_t_ph, axis=-1)
    target_q  = reward + done_mask * self.gamma * target_q
    target_q  = tf.stop_gradient(target_q)
    return target_q


  def _get_actor_loss(self, actor_critic_q):
    # Set the pseudo loss for the policy. Take the negative of the loss for Gradient Ascent
    return -tf.reduce_mean(actor_critic_q)


  def _get_critic_loss(self, target_q, agent_q):
    if self.huber_loss:
      critic_loss = tf.losses.huber_loss(target_q, agent_q)
    else:
      critic_loss = tf.losses.mean_squared_error(target_q, agent_q)
    if self.critic_reg != 0.0:
      critic_loss += tf.losses.get_regularization_loss(scope="agent_net/critic")

    return critic_loss


  def _group_control_deps(self, actor_grads, critic_grads):
    a_grad_tensors  = [grad for grad, var in actor_grads]
    c_grad_tensors  = [grad for grad, var in critic_grads]

    # Get the ops for updating the running mean and variance in batch norm layers
    batch_norm_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    control_deps    = a_grad_tensors + c_grad_tensors + batch_norm_ops
    logger.info("Control dependencies for apply_gradients():")
    for t in control_deps:
      logger.info(t.name)
    return control_deps


  def _build_train_op(self, actor_opt, critic_opt, actor_loss, critic_loss, actor_vars, critic_vars):
    actor_grads     = actor_opt.compute_gradients(actor_loss,   var_list=actor_vars)
    critic_grads    = critic_opt.compute_gradients(critic_loss, var_list=critic_vars)

    control_deps    = self._group_control_deps(actor_grads, critic_grads)

    # Apply gradients only after both actor and critic have been differentiated
    with tf.control_dependencies(control_deps):
      train_actor   = actor_opt.apply_gradients(actor_grads)
      train_critic  = critic_opt.apply_gradients(critic_grads)

    # Create train Op
    train_op  = tf.group(train_actor, train_critic, name="train_op")

    return train_op


  def initialize(self, sess):
    """Initialize the model. See Model.initialize()"""
    sess.run(self._init_op)


  def reset(self, sess):
    pass


  def action_train_ops(self, sess, state, run_dict=None):
    feed_dict = {self.obs_t_ph: state[None,:], self._training: False}
    return super()._action_train_ops(sess, run_dict, feed_dict=feed_dict)


  def action_eval_ops(self, sess, state, run_dict=None):
    feed_dict = {self.obs_t_ph: state[None,:], self._training: False}
    return super()._action_eval_ops(sess, run_dict, feed_dict=feed_dict)


  def _actor_net(self, state, scope):
    """Build actor network for low-dimensional observations
    Args:
      state: tf.Tensor. Input tensor for the state
      scope: string. Parent scope for the network variables
    Returns:
      `tf.Tensor` that holds the value of the control action
    """
    x = state
    n_actions = self.n_actions

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

      x = tf.layers.dense(x, 400, kernel_initializer=self.hidden_init(), name="dense1")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")
      x = tf.nn.relu(x)

      x = tf.layers.dense(x, 300, kernel_initializer=self.hidden_init(), name="dense2")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")
      x = tf.nn.relu(x)

      x = tf.layers.dense(x, n_actions, tf.nn.tanh, kernel_initializer=self.output_init(), name="dense3")

      return x


  def _critic_net(self, state, action, scope):
    """Build critic network for low-dimensional observations
    Args:
      state: tf.Tensor. Input tensor for the state. Batch must be the 0 dimension
      action: tf.Tensor. Input tensor for the action. Batch must be the 0 dimension
      scope: string. Parent scope for the network variables
    Returns:
      `tf.Tensor` that holds the value of the Q-function estimate
    """

    x = state
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      regularizer = tf.contrib.layers.l2_regularizer(scale=self.critic_reg)

      x = tf.layers.dense(x, 400, kernel_initializer=self.hidden_init(),
                          kernel_regularizer=regularizer, name="dense1")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")
      x = tf.nn.relu(x)

      x = tf.concat([x, action], axis=-1)

      # No batch norm after action input, as in the original paper
      x = tf.layers.dense(x, 300, kernel_initializer=self.hidden_init(),
                          kernel_regularizer=regularizer, name="dense2")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")
      x = tf.nn.relu(x)

      x = tf.layers.dense(x, 1, kernel_initializer=self.output_init(),
                          kernel_regularizer=regularizer, name="dense3")
      return x


  def _actor_conv_net(self, state, scope):
    """
    Args:
      state: tf.Tensor. Input tensor for the state
      scope: string. Parent scope for the network variables
    Returns:
      `tf.Tensor` that holds the value of the control action
    """

    x = state
    n_actions = self.n_actions

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv1")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv2")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv3")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm3")

      x = tf.layers.flatten(x)

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=self.hidden_init(), name="dense1")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm4")

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=self.hidden_init(), name="dense2")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm5")

      x = tf.layers.dense(x, n_actions, tf.nn.tanh, kernel_initializer=self.output_init(), name="dense3")

      return x


  def _critic_conv_net(self, state, action, scope):
    """Build critic network

    Args:
      state: tf.Tensor. Input tensor for the state. Batch must be the 0 dimension
      action: tf.Tensor. Input tensor for the action. Batch must be the 0 dimension
      scope: string. Parent scope for the network variables
    Returns:
      `tf.Tensor` that holds the value of the Q-function estimate
    """

    x = state
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv1")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv2")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv3")
      if self.batch_norm:
        x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm3")

      x = tf.layers.flatten(x)
      x = tf.concat([x, action], axis=-1)

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=self.hidden_init(), name="dense1")
      # x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm4")

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=self.hidden_init(), name="dense2")
      # x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm5")

      x = tf.layers.dense(x, 1, kernel_initializer=self.output_init())

      return x
