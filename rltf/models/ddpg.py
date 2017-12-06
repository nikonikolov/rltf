import numpy      as np
import tensorflow as tf

from rltf.models.model  import Model
from rltf.models        import tf_utils


def init_uniform():
  return tf.variance_scaling_initializer(scale=1.0/3.0, mode="fan_in", distribution="uniform")


class DDPG(Model):

  def __init__(self, obs_shape, act_min, act_max, actor_opt_conf, critic_opt_conf,
               critic_reg, tau, gamma, huber_loss=False):
    """
    Args:
      obs_shape: list. Shape of a single state input
      act_min: np.array. Minimum possible action values. Must have size n_actions
      act_max: np.array. Maximum possible action values. Must have size n_actions
      actor_opt_conf: rltf.optimizers.OptimizerConf. Configuration for the actor optimizer
      critic_opt_conf: rltf.optimizers.OptimizerConf. Configuration for the critic optimizer
      critic_reg: float. Regularization parameter for the weights of the critic
      tau: float. Rate of update for the target network
      gamma: float. Discount factor
      huber_loss: bool. Whether to use Huber loss or not
    """
    assert act_max.shape      == act_min.shape
    assert len(act_max.shape) == 1

    super().__init__()

    self.tau          = tau
    self.huber_loss   = huber_loss
    self.gamma        = gamma
    self.critic_reg   = critic_reg

    self.actor_opt_conf   = actor_opt_conf
    self.critic_opt_conf  = critic_opt_conf

    self.obs_shape = obs_shape
    self.obs_dtype = tf.uint8 if len(obs_shape) == 3 else tf.float32
    self.n_actions = act_max.size
    self.act_shape = [self.n_actions]
    self.act_dtype = tf.float32

    # Compute parameters to normalize and denormalize actions
    self._action_stats(act_min, act_max)

    # Custom TF Tensors and Ops
    self._init_op   = None
    self._training  = None


  def _action_stats(self, act_min, act_max):
    self.act_mean     = np.asarray((act_min + act_max) / 2.0, dtype=np.float32)
    act_std           = act_max - self.act_mean
    zeros             = (act_std <= 1e-5)
    act_std[zeros]    += 1e-4
    self.act_std      = np.asarray(act_std, dtype=np.float32)


  def build(self):

    super()._build()

    # Placehodler for the running mode - training or inference
    self._training      = tf.placeholder_with_default(True, (), name="training")
    # self._training      = tf.Variable(True, trainable=False)
    # self._set_train     = tf.assign(self._training, True,   name="set_train")
    # self._set_eval      = tf.assign(self._training, False,  name="set_eval")

    act_t_norm          = (self._act_t_ph - self.act_mean) / self.act_std

    # Conv net
    if self.obs_dtype == tf.uint8:
      obs_t_float   = tf.cast(self._obs_t_ph,   tf.float32) / 255.0
      obs_tp1_float = tf.cast(self._obs_tp1_ph, tf.float32) / 255.0

      actor   = self._actor_conv_net
      critic  = self._critic_conv_net

    # Low-dimensionsal net
    else:
      obs_t_float       = tf.layers.flatten(self._obs_t_ph)
      obs_tp1_float     = tf.layers.flatten(self._obs_tp1_ph)

      obs_t_float       = tf.layers.batch_normalization(obs_t_float, axis=-1,
                                                        training=self._training, trainable=False)
      obs_tp1_float     = tf.layers.batch_normalization(obs_tp1_float, axis=-1,
                                                        training=self._training, trainable=False)
      actor   = self._actor_net
      critic  = self._critic_net

    action          = actor(obs_t_float,                scope="agent_net/actor")
    actor_critic_q  = critic(obs_t_float, action,       scope="agent_net/critic")
    act_samples_q   = critic(obs_t_float, act_t_norm,   scope="agent_net/critic")

    target_act      = actor(obs_tp1_float,              scope="target_net/actor")
    target_q        = critic(obs_tp1_float, target_act, scope="target_net/critic")

    agent_vars      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_net")
    target_vars     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
    actor_vars      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_net/actor")
    critic_vars     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_net/critic")

    done_mask       = tf.cast(tf.logical_not(self._done_ph), tf.float32)
    target_q        = self._rew_t_ph + done_mask * self.gamma * target_q

    # Get Huber or L2 loss
    critic_tf_loss  = tf.losses.huber_loss if self.huber_loss else tf.losses.mean_squared_error

    # Set the pseudo loss for the policy. Take the negative of the loss for Gradient Ascent
    actor_loss      = -tf.reduce_mean(actor_critic_q)
    critic_loss     = critic_tf_loss(target_q, act_samples_q)
    critic_loss    += tf.losses.get_regularization_loss(scope="agent_net/critic")

    # Build the optimizers
    actor_opt       = self.actor_opt_conf.build()
    critic_opt      = self.critic_opt_conf.build()

    actor_grads     = actor_opt.compute_gradients(actor_loss,   var_list=actor_vars)
    critic_grads    = critic_opt.compute_gradients(critic_loss, var_list=critic_vars)

    a_grad_tensors  = [grad for grad, var in actor_grads]
    c_grad_tensors  = [grad for grad, var in critic_grads]

    # Get the ops for updating the running mean and variance in batch norm layers
    batch_norm_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    control_deps    = a_grad_tensors + c_grad_tensors + batch_norm_ops

    with tf.control_dependencies(control_deps):
      train_actor   = actor_opt.apply_gradients(actor_grads)
      train_critic  = critic_opt.apply_gradients(critic_grads)

    # Create train Op
    self._train_op  = tf.group(train_actor, train_critic, name="train_op")

    # Create the Op that updates the target
    self._update_target = tf_utils.assign_values(target_vars, agent_vars, self.tau, "update_target")

    # Remember the action tensor. name is needed when restoring the graph
    self._action    = tf.identity(action, name="action")
    # self._action    = tf.add(action * self.act_std, self.act_mean, name="action")

    # Initialization Op
    self._init_op   = tf_utils.assign_values(target_vars, agent_vars, name="init_op")

    # Summaries
    tf.summary.scalar("actor_loss",   actor_loss)
    tf.summary.scalar("critic_loss",  critic_loss)


  def _restore(self, graph):
    # Retrieve the Ops for changing between train and eval modes
    # self._set_train = graph.get_operation_by_name("set_train")
    # self._set_eval  = graph.get_operation_by_name("set_eval")
    self._training  = graph.get_tensor_by_name("training:0")
    self._init_op   = graph.get_operation_by_name("init_op")


  def initialize(self, sess):
    """Initialize the model. See Model.initialize()"""
    sess.run(self._init_op)


  def control_action(self, sess, state):
    feed_dict = {self._obs_t_ph: state[None,:], self._training: False}
    norm_act  = sess.run(self._action, feed_dict=feed_dict)[0]
    return norm_act * self.act_std + self.act_mean
    # return sess.run(self._action, feed_dict=feed_dict)[0]


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

      x = tf.layers.dense(x, 400, tf.nn.relu, kernel_initializer=init_uniform(), name="dense1")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")

      x = tf.layers.dense(x, 300, tf.nn.relu, kernel_initializer=init_uniform(), name="dense2")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")

      x = tf.layers.dense(x, n_actions, tf.nn.tanh, name="dense3",
                          kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

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

    regularizer = tf.contrib.layers.l2_regularizer(scale=self.critic_reg)
    x = state
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      x = tf.layers.dense(x, 400, tf.nn.relu, kernel_initializer=init_uniform(),
                          kernel_regularizer=regularizer, name="dense1")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")

      x = tf.concat([x, action], axis=-1)

      # No batch norm after action input, as in the original paper
      x = tf.layers.dense(x, 300, tf.nn.relu, kernel_initializer=init_uniform(),
                          kernel_regularizer=regularizer, name="dense2")

      x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(-3e-3,3e-3),
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
                           kernel_initializer=init_uniform(), name="conv1")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=init_uniform(), name="conv2")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=init_uniform(), name="conv3")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm3")

      x = tf.layers.flatten(x)

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=init_uniform(), name="dense1")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm4")

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=init_uniform(), name="dense2")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm5")

      x = tf.layers.dense(x, n_actions, tf.nn.tanh, name="dense3",
                          kernel_initializer=tf.random_uniform_initializer(-3e-4, 3e-4))

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
                           kernel_initializer=init_uniform(), name="conv1")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=init_uniform(), name="conv2")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")

      x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=init_uniform(), name="conv3")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm3")

      x = tf.layers.flatten(x)
      x = tf.concat([x, action], axis=-1)

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=init_uniform(), name="dense1")
      # x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm4")

      x = tf.layers.dense(x, 200, tf.nn.relu, kernel_initializer=init_uniform(), name="dense2")
      # x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm5")

      x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(-3e-4, 3e-4))

      return x
