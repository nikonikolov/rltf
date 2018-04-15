import tensorflow as tf

from rltf.models  import Model
from rltf.models  import tf_utils


class BaseDQN(Model):

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

    self.obs_shape  = obs_shape
    self.obs_dtype  = tf.uint8 if len(obs_shape) == 3 else tf.float32
    self.n_actions  = n_actions
    self.act_shape  = []
    self.act_dtype  = tf.uint8

    # Custom TF Tensors and Ops
    self.a_train    = None
    self.a_eval     = None
    self._obs_t     = None
    self._obs_tp1   = None


  def build(self):

    super()._build()

    # Preprocess the observation
    self._obs_t   = self._preprocess_obs(self._obs_t_ph)
    self._obs_tp1 = self._preprocess_obs(self._obs_tp1_ph)

    # Construct the Q-network and the target network
    agent_net     = self._nn_model(self._obs_t,   scope="agent_net")
    target_net    = self._nn_model(self._obs_tp1, scope="target_net")

    # Compute the estimated Q-function and its backup value
    estimate      = self._compute_estimate(agent_net)
    target        = self._compute_target(target_net)

    # Compute the loss
    loss          = self._compute_loss(estimate, target)

    agent_vars    = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_net")
    target_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")

    # Build the optimizer and the train op
    optimizer     = self.opt_conf.build()
    train_op      = self._build_train_op(optimizer, loss, agent_vars, name="train_op")

    # Create the Op to update the target
    update_target = tf_utils.assign_vars(target_vars, agent_vars, name="update_target")

    # Compute the train and eval actions
    self.a_train  = self._act_train(agent_net, name="a_train")
    self.a_eval   = self._act_eval(agent_net,  name="a_eval")

    self._train_op      = train_op
    self._update_target = update_target


  def _preprocess_obs(self, obs):
    if self.obs_dtype == tf.uint8:
      # In this case, casting on GPU ensures lower data transfer times
      return tf.cast(obs, tf.float32) / 255.0
    else:
      return obs


  def _nn_model(self, x, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      if len(self.obs_shape) == 3:
        return self._conv_nn(x)
      else:
        return self._dense_nn(x)


  def _conv_nn(self, x):
    raise NotImplementedError()


  def _dense_nn(self, x):
    raise NotImplementedError()


  def _act_train(self, agent_net, name):
    raise NotImplementedError()


  def _act_eval(self, agent_net, name):
    raise NotImplementedError()


  def _compute_estimate(self, agent_net):
    raise NotImplementedError()


  def _compute_target(self, target_net):
    raise NotImplementedError()


  def _compute_loss(self, estimate, target):
    raise NotImplementedError()


  def _build_train_op(self, optimizer, loss, agent_vars, name):
    train_op = optimizer.minimize(loss, var_list=agent_vars, name=name)
    return train_op


  def _restore(self, graph):
    # Get the train and eval action tensors
    self.a_train  = graph.get_tensor_by_name("a_train:0")
    self.a_eval   = graph.get_tensor_by_name("a_eval:0")


  def initialize(self, sess):
    """Initialize the model. See Model.initialize()"""
    sess.run(self._update_target)


  def reset(self, sess):
    pass


  def action_train(self, sess, state):
    assert list(state.shape) == self.obs_shape
    action = sess.run(self.a_train, feed_dict={self.obs_t_ph: state[None,:]})
    action = action[0]
    return action


  def action_eval(self, sess, state):
    assert list(state.shape) == self.obs_shape
    action = sess.run(self.a_eval, feed_dict={self.obs_t_ph: state[None,:]})
    action = action[0]
    return action


class DQN(BaseDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, huber_loss):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
      huber_loss: bool. Whether to use huber loss or not
    """

    super().__init__(obs_shape, n_actions, opt_conf, gamma)

    self.huber_loss = huber_loss


  def _conv_nn(self, x):
    """ Build the DQN architecture - as described in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created
    Returns:
      `tf.Tensor` of shape `[batch_size, n_actions]`. Contains the Q-function for each action
    """
    n_actions = self.n_actions

    with tf.variable_scope("conv_net"):
      # original architecture
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    with tf.variable_scope("action_value"):
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
      x = tf.layers.dense(x, units=n_actions, activation=None)
    return x


  def _dense_nn(self, x):
    """ Build a Neural Network of dense layers only. Used for low-level observations
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created
    Returns:
      `tf.Tensor` of shape `[batch_size, n_actions]`. Contains the Q-function for each action
    """
    n_actions = self.n_actions

    with tf.variable_scope("dense_net"):
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
      x = tf.layers.dense(x, units=n_actions, activation=None)
    return x


  def _act_train(self, agent_net, name):
    action = tf.argmax(agent_net, axis=-1, output_type=tf.int32, name=name)
    return action


  def _act_eval(self, agent_net, name):
    return tf.identity(self.a_train, name=name)


  def _compute_estimate(self, agent_net):
    # Get the Q value for the selected action; output shape [None]
    a_mask  = tf.one_hot(self._act_t_ph, self.n_actions, dtype=tf.float32)
    q       = tf.reduce_sum(agent_net * a_mask, axis=-1)

    return q


  def _compute_target(self, target_net):
    done_mask = tf.cast(tf.logical_not(self._done_ph), tf.float32)
    target_q  = tf.reduce_max(target_net, axis=-1)
    target_q  = self.rew_t_ph + self.gamma * done_mask * target_q
    target_q  = tf.stop_gradient(target_q)

    return target_q


  def _compute_loss(self, estimate, target):
    if self.huber_loss:
      loss = tf.losses.huber_loss(target, estimate)
    else:
      loss = tf.losses.mean_squared_error(target, estimate)
    tf.summary.scalar("train/loss", loss)
    return loss
