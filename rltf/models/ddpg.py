import logging

import tensorflow as tf

from rltf.models.model  import Model
from rltf.models        import tf_utils

logger = logging.getLogger(__name__)


def init_hidden_uniform():
  return tf.variance_scaling_initializer(scale=1.0/3.0, mode="fan_in", distribution="uniform",
                                         seed=rltf.conf.SEED)

def init_output_uniform():
  return tf.random_uniform_initializer(-3e-3, 3e-3, seed=rltf.conf.SEED)

def init_output_uniform_conv():
  return tf.random_uniform_initializer(-3e-4, 3e-4, seed=rltf.conf.SEED)


class DDPG(Model):

  def __init__(self, obs_shape, n_actions, actor_opt_conf, critic_opt_conf, critic_reg,
               tau, gamma, huber_loss=False):
    """
    Args:
      obs_shape: list. Shape of a single state input
      actor_opt_conf: rltf.optimizers.OptimizerConf. Configuration for the actor optimizer
      critic_opt_conf: rltf.optimizers.OptimizerConf. Configuration for the critic optimizer
      critic_reg: float. Regularization parameter for the weights of the critic
      tau: float. Rate of update for the target network
      gamma: float. Discount factor
      huber_loss: bool. Whether to use Huber loss or not
    """

    super().__init__()

    self.tau          = tau
    self.huber_loss   = huber_loss
    self.gamma        = gamma
    self.critic_reg   = critic_reg

    self.actor_opt_conf   = actor_opt_conf
    self.critic_opt_conf  = critic_opt_conf

    self.obs_shape = obs_shape
    self.obs_dtype = tf.uint8 if len(obs_shape) == 3 else tf.float32
    self.n_actions = n_actions
    self.act_shape = [self.n_actions]
    self.act_dtype = tf.float32

    # Initializers
    self.hidden_init = init_hidden_uniform
    self.output_init = init_output_uniform

    # Custom TF Tensors and Ops
    self._init_op   = None
    self._training  = None


  def build(self):

    super()._build()

    # Placehodler for the running mode - training or evaluation
    self._training      = tf.placeholder_with_default(True, (), name="training")

    # Conv net
    if self.obs_dtype == tf.uint8:
      # Normalize observations
      obs_t_float   = tf.cast(self._obs_t_ph,   tf.float32) / 255.0
      obs_tp1_float = tf.cast(self._obs_tp1_ph, tf.float32) / 255.0

      actor   = self._actor_conv_net
      critic  = self._critic_conv_net

    # Low-dimensionsal net
    else:
      obs_t_float       = tf.layers.flatten(self._obs_t_ph)
      obs_tp1_float     = tf.layers.flatten(self._obs_tp1_ph)

      # Normalize observations
      obs_t_float       = tf.layers.batch_normalization(obs_t_float, axis=-1, trainable=False,
                                    center=False, scale=False, training=self._training)
      obs_tp1_float     = tf.layers.batch_normalization(obs_tp1_float, axis=-1, trainable=False,
                                    center=False, scale=False, training=self._training)
      actor   = self._actor_net
      critic  = self._critic_net

    action          = actor(obs_t_float,                  scope="agent_net/actor")
    actor_critic_q  = critic(obs_t_float, action,         scope="agent_net/critic")
    act_t_q         = critic(obs_t_float, self._act_t_ph, scope="agent_net/critic")

    target_act      = actor(obs_tp1_float,                scope="target_net/actor")
    target_q        = critic(obs_tp1_float, target_act,   scope="target_net/critic")

    agent_vars      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_net")
    target_vars     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
    actor_vars      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_net/actor")
    critic_vars     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_net/critic")

    target_q        = self._compute_target(target_q)

    # Set the pseudo loss for the policy. Take the negative of the loss for Gradient Ascent
    actor_loss      = self._get_actor_loss(actor_critic_q)
    critic_loss     = self._get_critic_loss(target_q, act_t_q)

    # Build the optimizers
    actor_opt       = self.actor_opt_conf.build()
    critic_opt      = self.critic_opt_conf.build()

    actor_grads     = actor_opt.compute_gradients(actor_loss,   var_list=actor_vars)
    critic_grads    = critic_opt.compute_gradients(critic_loss, var_list=critic_vars)

    control_deps    = self._group_control_deps(actor_grads, critic_grads)

    # Apply gradients only after both actor and critic have been differentiated
    with tf.control_dependencies(control_deps):
      train_actor   = actor_opt.apply_gradients(actor_grads)
      train_critic  = critic_opt.apply_gradients(critic_grads)

    # Create train Op
    self._train_op  = tf.group(train_actor, train_critic, name="train_op")

    # Create the Op that updates the target
    logger.debug("Creating target net update Op")
    self._update_target = tf_utils.assign_vars(target_vars, agent_vars, self.tau, "update_target")

    # Remember the action tensor. name is needed when restoring the graph
    self._action    = tf.identity(action, name="action")

    # Initialization Op
    logger.debug("Creating initialization Op")
    self._init_op   = tf_utils.assign_vars(target_vars, agent_vars, name="init_op")

    # Summaries
    tf.summary.scalar("actor_loss",   actor_loss)
    tf.summary.scalar("critic_loss",  critic_loss)

    tf.summary.scalar("actor_critic_q", -actor_loss)
    tf.summary.scalar("act_t_q",        tf.reduce_mean(act_t_q))
    tf.summary.scalar("target_q",       tf.reduce_mean(target_q))


  def _compute_target(self, target_q):
    done_mask = tf.cast(tf.logical_not(self._done_ph), tf.float32)
    done_mask = tf.expand_dims(done_mask, axis=-1)
    reward    = tf.expand_dims(self._rew_t_ph, axis=-1)
    return reward + done_mask * self.gamma * target_q


  def _get_actor_loss(self, actor_critic_q):
    return -tf.reduce_mean(actor_critic_q)


  def _get_critic_loss(self, target_q, agent_q):
    if self.huber_loss:
      critic_loss = tf.losses.huber_loss(target_q, agent_q)
    else:
      critic_loss = tf.losses.mean_squared_error(target_q, agent_q)
    critic_loss    += tf.losses.get_regularization_loss(scope="agent_net/critic")

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


  def _restore(self, graph):
    # Retrieve the Ops for changing between train and eval modes
    self._training  = graph.get_tensor_by_name("training:0")
    self._init_op   = graph.get_operation_by_name("init_op")


  def initialize(self, sess):
    """Initialize the model. See Model.initialize()"""
    sess.run(self._init_op)


  def control_action(self, sess, state):
    feed_dict = {self._obs_t_ph: state[None,:], self._training: False}
    return sess.run(self._action, feed_dict=feed_dict)[0]


  def _actor_net(self, state, scope):
    """
    Args:
      state: tf.Tensor. Input tensor for the state
      scope: string. Parent scope for the network variables. Must end in "/"
    Returns:
      `tf.Tensor` that holds the value of the control action
    """
    x = state
    n_actions = self.n_actions

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

      x = tf.layers.dense(x, 400, kernel_initializer=self.hidden_init(), name="dense1")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")
      x = tf.nn.relu(x)

      x = tf.layers.dense(x, 300, kernel_initializer=self.hidden_init(), name="dense2")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")
      x = tf.nn.relu(x)

      x = tf.layers.dense(x, n_actions, tf.nn.tanh, kernel_initializer=self.output_init(), name="dense3")

      return x


  def _critic_net(self, state, action, scope):
    """Build critic network

    Args:
      state: tf.Tensor. Input tensor for the state. Batch must be the 0 dimension
      action: tf.Tensor. Input tensor for the action. Batch must be the 0 dimension
      scope: string. Parent scope for the network variables. Must end in "/"
    Returns:
      `tf.Tensor` that holds the value of the Q-function estimate
    """

    x = state
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      regularizer = tf.contrib.layers.l2_regularizer(scale=self.critic_reg)

      x = tf.layers.dense(x, 400, kernel_initializer=self.hidden_init(),
                          kernel_regularizer=regularizer, name="dense1")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")
      x = tf.nn.relu(x)

      x = tf.concat([x, action], axis=-1)

      # No batch norm after action input, as in the original paper
      x = tf.layers.dense(x, 300, kernel_initializer=self.hidden_init(),
                          kernel_regularizer=regularizer, name="dense2")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")
      x = tf.nn.relu(x)

      x = tf.layers.dense(x, 1, kernel_initializer=self.output_init(),
                          kernel_regularizer=regularizer, name="dense3")
      return x


  def _actor_conv_net(self, state, scope):
    """
    Args:
      state: tf.Tensor. Input tensor for the state
      scope: string. Parent scope for the network variables. Must end in "/"
    Returns:
      `tf.Tensor` that holds the value of the control action
    """

    x = state
    n_actions = self.n_actions

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv1")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv2")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv3")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm3")

      x = tf.layers.flatten(x)

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=self.hidden_init(), name="dense1")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm4")

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=self.hidden_init(), name="dense2")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm5")

      x = tf.layers.dense(x, n_actions, tf.nn.tanh, kernel_initializer=self.output_init(), name="dense3")

      return x


  def _critic_conv_net(self, state, action, scope):
    """Build critic network

    Args:
      state: tf.Tensor. Input tensor for the state. Batch must be the 0 dimension
      action: tf.Tensor. Input tensor for the action. Batch must be the 0 dimension
      scope: string. Parent scope for the network variables. Must end in "/"
    Returns:
      `tf.Tensor` that holds the value of the Q-function estimate
    """

    x = state
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv1")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv2")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=self.hidden_init(), name="conv3")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm3")

      x = tf.layers.flatten(x)
      x = tf.concat([x, action], axis=-1)

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=self.hidden_init(), name="dense1")
      # x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm4")

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=self.hidden_init(), name="dense2")
      # x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm5")

      x = tf.layers.dense(x, 1, kernel_initializer=self.output_init())

      return x
