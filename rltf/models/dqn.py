import numpy      as np
import tensorflow as tf

from rltf.models.model  import Model
from rltf.models        import tf_utils


class BaseDQN(Model):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
    """

    super().__init__()

    self.gamma      = gamma
    self.opt_conf   = opt_conf

    self.obs_shape  = obs_shape
    self.obs_dtype  = tf.uint8
    self.n_actions  = n_actions
    self.act_shape  = []
    self.act_dtype  = tf.uint8

    # Custom TF Tensors and Ops
    self._q         = None


  def build(self):

    super()._build()
    # Input placeholders
    # self._obs_t_ph    = tf.placeholder(tf.uint8,   [None] + self.obs_shape, name="obs_t_ph")
    # self._act_t_ph    = tf.placeholder(tf.uint8,   [None],                  name="act_t_ph")
    # self._rew_t_ph    = tf.placeholder(tf.float32, [None],                  name="rew_t_ph")
    # self._obs_tp1_ph  = tf.placeholder(tf.uint8,   [None] + self.obs_shape, name="obs_tp1_ph")
    # self._done_ph     = tf.placeholder(tf.bool,    [None],                  name="done_ph")

    # In this case, casting on GPU ensures lower data transfer times
    obs_t       = tf.cast(self._obs_t_ph,   tf.float32) / 255.0
    obs_tp1     = tf.cast(self._obs_tp1_ph, tf.float32) / 255.0

    # Construct the Q-network and the target network
    estimate    = self._nn_model(obs_t,   scope="agent_net")
    target      = self._nn_model(obs_tp1, scope="target_net")

    # Save the Q-function estimate tensor
    self._q     = tf.identity(self._compute_q(estimate), name="q_fn")

    # Compute the estimated Q-function and its backup value
    estimate    = self._compute_estimate(estimate)
    target      = self._compute_target(target)

    # Compute the loss
    loss        = self._compute_loss(estimate, target)

    agent_vars  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_net')
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

    # Build the optimizer
    optimizer   = self.opt_conf.build()
    # Create the training Op
    train_op    = optimizer.minimize(loss, var_list=agent_vars, name="train_op")
    # Create the Op to update the target
    target_op   = tf_utils.assign_vars(target_vars, agent_vars, name="update_target")

    self._train_op      = train_op
    self._update_target = target_op

    # Add summaries
    tf.summary.scalar("loss", loss)


  def _nn_model(self, x, scope):
    raise NotImplementedError()


  def _compute_q(self, nn_out):
    raise NotImplementedError()


  def _compute_estimate(self, nn_out):
    raise NotImplementedError()


  def _compute_target(self, nn_out):
    raise NotImplementedError()


  def _compute_loss(self, estimate, target):
    raise NotImplementedError()


  def _restore(self, graph):
    # Get Q-function Tensor
    self._q = graph.get_tensor_by_name("q_fn:0")


  def initialize(self, sess):
    """Initialize the model. See Model.initialize()"""
    sess.run(self._update_target)


  def reset(self, sess):
    pass


  def control_action(self, sess, state):
    q_vals  = sess.run(self._q, feed_dict={self.obs_t_ph: state[None,:]})
    action  = np.argmax(q_vals)

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


  def _nn_model(self, x, scope):
    """ Build the DQN architecture - as described in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created

    Returns:
      `tf.Tensor` of shape `[batch_size, n_actions]`. Contains the Q-function for each action
    """
    n_actions = self.n_actions

    with tf.variable_scope(scope, reuse=False):
      with tf.variable_scope("convnet"):
        # original architecture
        x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.flatten(x)
      with tf.variable_scope("action_value"):
        x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
        x = tf.layers.dense(x, units=n_actions, activation=None)
      return x


  def _compute_q(self, nn_out):
    return nn_out


  def _compute_estimate(self, nn_out):
    # Get the Q value for the selected action; output shape [None]
    q         = nn_out
    act_t     = tf.cast(self._act_t_ph, tf.int32)
    act_mask  = tf.one_hot(act_t, self.n_actions, on_value=True, off_value=False, dtype=tf.bool)
    q         = tf.boolean_mask(q, act_mask)

    return q


  def _compute_target(self, nn_out):
    target_q  = nn_out
    done_mask = tf.cast(tf.logical_not(self._done_ph), tf.float32)
    target_q  = tf.reduce_max(target_q, axis=-1)
    target_q  = self.rew_t_ph + self.gamma * done_mask * target_q

    return target_q


  def _compute_loss(self, estimate, target):
    loss_fn   = tf.losses.huber_loss if self.huber_loss else tf.losses.mean_squared_error
    loss      = loss_fn(target, estimate)

    return loss
