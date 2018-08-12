import numpy as np
import tensorflow as tf

from rltf.models.bstrap_dqn import BaseBstrapDQN
from rltf.models.bstrap_dqn import BstrapDQN_IDS
from rltf.models.c51        import C51
from rltf.models            import tf_utils


class BaseBstrapC51(BaseBstrapDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, n_heads, V_min, V_max, N):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
      n_heads: Number of bootstrap heads
      V_min: float. lower bound for histrogram range
      V_max: float. upper bound for histrogram range
      N: int. number of histogram bins
    """
    super().__init__(obs_shape, n_actions, opt_conf, gamma, False, n_heads)

    self.N      = N
    self.V_min  = V_min
    self.V_max  = V_max
    self.dz     = (self.V_max - self.V_min) / float(self.N - 1)

    # Custom TF Tensors and Ops
    self.bins   = None


  def build(self):
    # Costruct the tensor of the bins for the probability distribution
    bins      = np.arange(0, self.N, 1, dtype=np.float32)
    bins      = bins * self.dz + self.V_min
    self.bins = tf.constant(bins[None, None, :], dtype=tf.float32)  # out shape: [1, 1, N]

    super().build()


  def _conv_nn(self, x):
    """ Build the Bootstrapped DQN architecture - as described in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
    Returns:
      Tuple of `tf.Tensor`s. First tensor is of shape `[batch_size, n_heads, n_actions]` and contains the
      Q-function bootstrapped estimates. Second tensor is of shape `[batch_size, n_actions, N]` and
      contains the C51 return distribution for each head
    """
    n_actions = self.n_actions
    N         = self.N

    def build_bstrap_head(x):
      """ Build the head of the DQN network
      Args:
        x: tf.Tensor. Tensor for the input
      Returns:
        `tf.Tensor` of shape `[batch_size, 1, n_actions]`. Contains the Q-function for each action
      """
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
      x = tf.layers.dense(x, units=n_actions, activation=None)
      x = tf.expand_dims(x, axis=-2)
      return x

    def build_z_head(x):
      """ Build the head of the C51 network
      Args:
        x: tf.Tensor. Tensor for the input
      Returns:
        `tf.Tensor` of shape `[batch_size, n_actions, N]`. Contains the Q-function distribution
          for each action
      """
      x = tf.layers.dense(x, 512,         activation=tf.nn.relu)
      x = tf.layers.dense(x, N*n_actions, activation=None)
      x = tf.reshape(x, [-1, n_actions, N])
      # Compute Softmax probabilities in numerically stable way
      C = tf.stop_gradient(tf.reduce_max(x, axis=-1, keepdims=True))
      x = tf.nn.softmax(x-C, axis=-1)
      return x

    with tf.variable_scope("conv_net"):
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    x = tf.layers.flatten(x)

    # Careful: Make sure self._conv_out is set only during the right function call
    if "agent_net" in tf.get_variable_scope().name and self._conv_out is None: self._conv_out = x

    # Build the C51 head
    with tf.variable_scope("distribution_value"):
      z = build_z_head(x)

    # Build the Bootstrap heads
    with tf.variable_scope("action_value"):
      heads = [build_bstrap_head(x) for _ in range(self.n_heads)]
      x = tf.concat(heads, axis=-2)
    return x, z


  def _compute_estimate(self, agent_net):
    """Get the Q value for the selected action
    Args:
      agent_net: tuple of `tf.Tensor`s. Output from the agent network. Shapes:
        `[batch_size, n_heads, n_actions]` and `[batch_size, n_actions, N]`
    Returns:
      Tuple of `tf.Tensor`s of shapes `[batch_size, n_heads]` and `[batch_size, N]`
    """
    q, z = agent_net
    q = super()._compute_estimate(q)
    z = C51._compute_estimate(self, z)
    # a_mask  = tf.one_hot(self._act_t_ph, self.n_actions, dtype=tf.float32)  # out: [None, n_actions]
    # a_mask  = tf.expand_dims(a_mask, axis=-1)                               # out: [None, n_actions, 1]
    # z       = tf.reduce_sum(z * a_mask, axis=1)                             # out: [None, N]
    return q, z


  def _select_target(self, target_net):
    """Select the Double DQN target
    Args:
      target_net: `tf.Tensor`. shape `[None, n_heads, n_actions]. The output from `self._nn_model()`
        for the target
    Returns:
      `tf.Tensor` of shape `[None, n_heads]`
    """
    n_actions   = self.n_actions

    # Compute the Q-estimate with the agent network variables and select the maximizing action
    agent_net   = self._nn_model(self._obs_tp1, scope="agent_net")      # out: [None, n_heads, n_actions]
    agent_net   = agent_net[0]  # Select only the Q-tensor
    target_act  = tf.argmax(agent_net, axis=-1, output_type=tf.int32)   # out: [None, n_heads]

    # Select the target Q-function
    target_mask = tf.one_hot(target_act, n_actions, dtype=tf.float32)   # out: [None, n_heads, n_actions]
    target_q    = tf.reduce_sum(target_net * target_mask, axis=-1)      # out: [None, n_heads]

    return target_q


  def _compute_target(self, target_net):
    """Compute the backups
    Args:
      target_net: tuple of `tf.Tensor`s. Output from the target network. Shapes:
        `[batch_size, n_heads, n_actions]` and `[batch_size, n_actions, N]`
    Returns:
      Tuple of `tf.Tensor`s of shapes `[batch_size, n_heads]` and `[batch_size, N]`
    """
    target_q, target_z = target_net
    backup_q = super()._compute_target(target_q)
    # backup_z = C51._compute_target(self, target_z)  # NOT correct
    target_z = C51._select_target(self, target_z)
    backup_z = C51._compute_backup(self, target_z)
    backup_z = tf.stop_gradient(backup_z)
    return backup_q, backup_z


  def _compute_loss(self, estimate, target, name):
    estimate_q, z       = estimate
    target_q, target_z  = target

    head_loss = super()._compute_loss(estimate_q, target_q, name)

    z_loss    = C51._compute_loss(self, z, target_z, "train/z_loss")
    # entropy   = -tf.reduce_sum(target_z * tf.log(z), axis=-1)
    # z_loss    = tf.reduce_mean(entropy)
    # tf.summary.scalar("train/z_loss", z_loss)

    return head_loss, z_loss


  def _build_train_op(self, optimizer, loss, agent_vars, name):
    head_loss = loss[0]
    z_loss    = loss[1]

    # Get the Bootsrapped heads and conv net train op
    train_net = super()._build_train_op(optimizer, head_loss, agent_vars, name=None)

    # Build the train op for C51 - apply gradients only to fully connected layers
    z_vars    = tf_utils.scope_vars(agent_vars, scope='agent_net/distribution_value')
    train_z   = optimizer.minimize(z_loss, var_list=z_vars)
    train_op  = tf.group(train_net, train_z, name=name)
    return train_op


  def _compute_z_variance(self, agent_net):
    z       = agent_net[1]

    # Var(X) = sum_x p(X)*[X - E[X]]^2
    q       = tf.reduce_sum(z * self.bins, axis=-1)       # out: [None, n_actions]
    center  = self.bins - tf.expand_dims(q, axis=-1)      # out: [None, n_actions, N]
    z_var   = tf.square(center) * z                       # out: [None, n_actions, N]
    z_var   = tf.reduce_sum(z_var, axis=-1)               # out: [None, n_actions]

    # Normalize the variance
    mean    = tf.reduce_mean(z_var, axis=-1, keepdims=True)   # out: [None, 1]
    z_var   = z_var / mean                                    # out: [None, n_actions]
    return z_var



class BstrapC51_IDS(BaseBstrapC51):
  """IDS policy from Boostrapped DQN-C51"""

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, n_heads, V_min, V_max, N, n_stds=0.1):
    super().__init__(obs_shape, n_actions, opt_conf, gamma, n_heads, V_min, V_max, N)

    self.n_stds = n_stds    # Number of standard deviations for computing uncertainty
    self.rho2   = None


  def _act_train(self, agent_net, name):
    # agent_net tuple of shapes: [None, n_heads, n_actions], [None, n_actions, N]

    z_var     = self._compute_z_variance(agent_net)           # out: [None, n_actions]
    self.rho2 = tf.maximum(z_var, 0.25)

    # Compute the IDS action - ugly way
    action    = BstrapDQN_IDS._act_train(self, agent_net[0], name)

    # Add debugging data for TB
    tf.summary.histogram("debug/a_rho2", self.rho2)
    tf.summary.scalar("debug/z_var", tf.reduce_mean(z_var))

    # p_rho2  = tf.identity(self.z_var[0], name="plot/train/rho2")
    p_rho2  = tf.identity(self.rho2[0], name="plot/train/rho2")
    p_a     = self.plot_train["train_actions"]["a_mean"]["a"]

    self.plot_train["train_actions"]["a_rho2"] = dict(height=p_rho2,  a=p_a)

    return action


  def _act_eval(self, agent_net, name):
    q = agent_net[0]
    return self._act_eval_greedy(q, name)
