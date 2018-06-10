import numpy as np
import tensorflow as tf

from rltf.models.bstrap_dqn import BaseBstrapDQN
from rltf.models.bstrap_dqn import BstrapDQN_IDS


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
      `tf.Tensor` of shape `[batch_size, n_heads, n_actions, N]`. Contains the Q-function distribution for
      each action in every head
    """
    n_actions = self.n_actions
    N         = self.N

    def build_head(x):
      """ Build the head of the C51 network
      Args:
        x: tf.Tensor. Tensor for the input
      Returns:
        `tf.Tensor` of shape `[batch_size, 1, n_actions, N]`. Contains the Q-function distribution
          for each action
      """
      x = tf.layers.dense(x, 512,         activation=tf.nn.relu)
      x = tf.layers.dense(x, N*n_actions, activation=None)
      x = tf.reshape(x, [-1, n_actions, N])
      # Compute Softmax probabilities in numerically stable way
      x = x - tf.expand_dims(tf.reduce_max(x, axis=-1), axis=-1)
      x = tf.nn.softmax(x, axis=-1)
      x = tf.expand_dims(x, axis=1)
      return x

    with tf.variable_scope("conv_net"):
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    x = tf.layers.flatten(x)

    # Careful: Make sure self._conv_out is set only during the right function call
    if "agent_net" in tf.get_variable_scope().name and self._conv_out is None: self._conv_out = x

    with tf.variable_scope("action_value"):
      heads = [build_head(x) for _ in range(self.n_heads)]
      x = tf.concat(heads, axis=1)
    return x


  def _compute_estimate(self, agent_net):
    """Get the Q value for the selected action
    Args:
      agent_net: tf.Tensor. Output from the network. shape: [batch_size, n_heads, n_actions, N]
    Returns:
      `tf.Tensor` of shape `[None, n_heads, N]`
    """
    a_mask  = tf.one_hot(self._act_t_ph, self.n_actions, dtype=tf.float32)    # out: [None, n_actions]
    a_mask  = tf.reshape(a_mask, [-1, 1, self.n_actions, 1])    # out: [None, 1,       n_actions]
    a_mask  = tf.tile(a_mask, [1, self.n_heads, 1, 1])          # out: [None, n_heads, n_actions, 1]
    z       = tf.reduce_sum(agent_net * a_mask, axis=-2)        # out: [None, n_heads, N]
    return z


  def _select_target(self, target_net):
    """Select the C51 target distributions for each head - use the greedy action from E[Z]
    Args:
      target_net: `tf.Tensor`, shape `[None, n_heads, n_actions, N]. The tensor output from
        `self._nn_model()` for the target
    Returns:
      `tf.Tensor` of shape `[None, N]`
    """
    n_actions   = self.n_actions
    target_z    = target_net

    # Compute the target q function and the greedy action
    bins        = tf.expand_dims(self.bins, axis=0)                     # out: [1, 1, 1, N]
    target_q    = tf.reduce_sum(target_z * bins, axis=-1)               # out: [None, n_heads, n_actions]
    target_act  = tf.argmax(target_q, axis=-1, output_type=tf.int32)    # out: [None, n_heads]
    target_mask = tf.one_hot(target_act, n_actions, dtype=tf.float32)   # out: [None, n_heads, n_actions]
    target_mask = tf.expand_dims(target_mask, axis=-1)                  # out: [None, n_heads, n_actions, 1]
    target_z    = tf.reduce_sum(target_net * target_mask, axis=-2)      # out: [None, n_heads, N]
    return target_z


  def _compute_backup(self, target):
    """Compute the C51 backup distributions
    Args:
      target: `tf.Tensor`, shape `[None, n_heads, N]. The output from `self._select_target()`
    Returns:
      `tf.Tensor` of shape `[None, n_heads, N]`
    """

    def build_inds_tensors(bin_inds_lo, bin_inds_hi):
      batch       = tf.shape(self.done_ph)[0]

      batch_inds  = tf.range(0, limit=batch, delta=1, dtype=tf.int32)
      batch_inds  = tf.reshape(batch_inds, [-1, 1, 1])                # out: [None, n_heads, 1]
      batch_inds  = tf.tile(batch_inds, [1, self.n_heads, self.N])    # out: [None, n_heads, N]
      batch_inds  = tf.expand_dims(batch_inds, axis=-1)               # out: [None, n_heads, N, 1]

      head_inds   = tf.range(0, limit=self.n_heads, delta=1, dtype=tf.int32)
      head_inds   = tf.reshape(head_inds, [1, -1, 1])                 # out: [1,    n_heads, 1]
      head_inds   = tf.tile(head_inds, [batch, 1, self.N])            # out: [None, n_heads, N]
      head_inds   = tf.expand_dims(head_inds, axis=-1)                # out: [None, n_heads, N, 1]

      bin_inds_lo = tf.expand_dims(tf.to_int32(bin_inds_lo), axis=-1) # out: [None, n_heads, N, 1]
      bin_inds_hi = tf.expand_dims(tf.to_int32(bin_inds_hi), axis=-1) # out: [None, n_heads, N, 1]
      bin_inds_lo = tf.concat([batch_inds, head_inds, bin_inds_lo], axis=-1)  # out: [None, n_heads, N, 3]
      bin_inds_hi = tf.concat([batch_inds, head_inds, bin_inds_hi], axis=-1)  # out: [None, n_heads, N, 3]

      return bin_inds_lo, bin_inds_hi

    target_z    = target

    # Compute projected bin support
    done_mask   = tf.cast(tf.logical_not(self.done_ph), tf.float32)     # out: [None]
    done_mask   = tf.expand_dims(done_mask, axis=-1)                    # out: [None, 1]
    rew_t       = tf.expand_dims(self.rew_t_ph, axis=-1)                # out: [None, 1]
    bins        = tf.squeeze(self.bins, axis=0)                         # out: [None, N]
    target_bins = rew_t + self.gamma * done_mask * bins                 # out: [None, N]
    target_bins = tf.clip_by_value(target_bins, self.V_min, self.V_max) # out: [None, N]
    target_bins = tf.expand_dims(target_bins, axis=1)                   # out: [None, 1, N]
    target_bins = tf.tile(target_bins, [1, self.n_heads, 1])            # out: [None, n_heads, N]

    # Projected bin indices
    bin_inds    = (target_bins - self.V_min) / self.dz  # out: [None, n_heads, N]
    bin_inds_lo = tf.floor(bin_inds)                    # out: [None, n_heads, N]
    bin_inds_hi = tf.ceil(bin_inds)                     # out: [None, n_heads, N]

    lo_add      = target_z * (bin_inds_hi - bin_inds)   # out: [None, n_heads, N]
    hi_add      = target_z * (bin_inds - bin_inds_lo)   # out: [None, n_heads, N]

    # Initialize the Variable holding the target distribution - gets reset to 0 every time
    zeros       = tf.zeros_like(bin_inds, dtype=tf.float32)
    target_z    = tf.Variable(0, trainable=False, dtype=tf.float32, validate_shape=False)
    target_z    = tf.assign(target_z, zeros, validate_shape=False)    # out: [None, 1, N]

    # Compute indices for scatter_nd_add
    inds        = build_inds_tensors(bin_inds_lo, bin_inds_hi)
    bin_inds_lo = inds[0]                               # out: [None, n_heads, N, 3]
    bin_inds_hi = inds[1]                               # out: [None, n_heads, N, 3]

    with tf.control_dependencies([target_z]):
      target_z  = tf.scatter_nd_add(target_z, bin_inds_lo, lo_add, use_locking=True)
      target_z  = tf.scatter_nd_add(target_z, bin_inds_hi, hi_add, use_locking=True)

    return target_z


  def _compute_loss(self, estimate, target):
    """
    Args:
      estimate: tf.Tensor, shape `[None, n_heads, N]`. Q-function estimate
      target: tf.Tensor, shape `[None, n_heads, N]`. Q-function target
    Returns:
      List of size `n_heads` with a scalar tensor loss for each head
    """

    estimates     = tf.split(estimate, self.n_heads, axis=1)    # list of shapes [None, 1, N]
    targets       = tf.split(target,   self.n_heads, axis=1)    # list of shapes [None, 1, N]
    estimates     = [tf.squeeze(e, axis=1) for e in estimates]  # list of shapes [None, N]
    targets       = [tf.squeeze(t, axis=1) for t in targets]    # list of shapes [None, N]

    def compute_head_loss(z, target_z):
      entropy = -tf.reduce_sum(target_z * tf.log(z), axis=-1)
      loss    = tf.reduce_mean(entropy)
      return loss

    losses = [compute_head_loss(z, target_z) for z, target_z in zip(estimates, targets)]

    tf.summary.scalar("train/loss", tf.add_n(losses)/self.n_heads)

    return losses


  def _compute_z_variance(self, agent_net):
    # Var(X) = sum_x p(X)*[X - E[X]]^2
    bins    = tf.expand_dims(self.bins, axis=0)
    z_mean  = tf.reduce_sum(agent_net * bins, axis=-1)    # out: [None, n_heads, n_actions]
    center  = bins - tf.expand_dims(z_mean, axis=-1)      # out: [None, n_heads, n_actions, N]
    z_var   = tf.square(center) * agent_net               # out: [None, n_heads, n_actions, N]
    z_var   = tf.reduce_sum(z_var, axis=-1)               # out: [None, n_heads, n_actions]

    # Take the sample mean of all heads
    # z_var   = tf.reduce_mean(z_var, axis=1)               # out: [None, n_actions]
    z_var   = tf.reduce_sum(z_var, axis=1) / float(self.n_heads-1)  # out: [None, n_actions]

    # Normalize the variance
    a_var   = tf.reduce_sum(tf.square(z_var), axis=-1)    # out: [None]
    a_var   = tf.expand_dims(tf.sqrt(a_var), axis=-1)     # out: [None, 1]
    z_var   = z_var / a_var                               # out: [None, n_actions]
    return z_var



class BstrapC51(BaseBstrapC51):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, n_heads, V_min, V_max, N):
    super().__init__(obs_shape, n_actions, opt_conf, gamma, n_heads, V_min, V_max, N)

    # Custom TF Tensors and Ops
    self._active_head   = None
    self._set_act_head  = None


  def build(self):
    self._active_head   = tf.Variable([0], trainable=False, name="active_head")
    sample_head         = tf.random_uniform(shape=[1], maxval=self.n_heads, dtype=tf.int32)
    self._set_act_head  = tf.assign(self._active_head, sample_head, name="set_act_head")
    super().build()


  def _act_train(self, agent_net, name):
    """Select the greedy action from the selected head based on E[Z]
    Args:
      agent_net: `tf.Tensor`, shape `[None, n_heads, n_actions, N]. The tensor output from
        `self._nn_model()` for the agent
    Returns:
      `tf.Tensor` of shape `[None]`
    """

    # Add debug info about the variance of the return
    z_var     = self._compute_z_variance(agent_net)
    tf.summary.scalar("debug/z_var", tf.reduce_mean(z_var))

    # Get the Z distribution from the active head
    head_mask = tf.one_hot(self._active_head, self.n_heads, dtype=tf.float32) # out: [1,    n_heads]
    head_mask = tf.tile(head_mask, [tf.shape(agent_net)[0], 1])               # out: [None, n_heads]
    head_mask = tf.reshape(head_mask, [-1, self.n_heads, 1, 1])               # out: [None, n_heads, 1, 1]
    z         = tf.reduce_sum(agent_net * head_mask, axis=1)                  # out: [None, n_actions, N]

    # Compute the greedy action
    q       = tf.reduce_sum(z * self.bins, axis=-1)                           # out: [None, n_actions]
    action  = tf.argmax(q, axis=-1, output_type=tf.int32, name=name)          # out: [None]
    return action


  def _act_eval(self, agent_net, name):
    bins = tf.expand_dims(self.bins, axis=0)      # out: [1, 1, 1, N]
    q = tf.reduce_sum(agent_net * bins, axis=-1)  # out: [None, n_heads, n_actions]
    return self._act_eval_vote(q, name)


  def reset(self, sess):
    sess.run(self._set_act_head)



class BstrapC51_IDS(BaseBstrapC51):
  """IDS policy from Boostrapped QRDQN"""

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, n_heads, V_min, V_max, N, policy, n_stds=0.1):
    super().__init__(obs_shape, n_actions, opt_conf, gamma, n_heads, V_min, V_max, N)

    assert policy in ["stochastic", "deterministic"]
    self.n_stds = n_stds    # Number of standard deviations for computing uncertainty
    self.policy = policy
    self.rho2   = None


  def _act_train(self, agent_net, name):
    # agent_net shape: [None, n_heads, n_actions, N]

    bins      = tf.expand_dims(self.bins, axis=0)
    q_heads   = tf.reduce_sum(agent_net * bins, axis=-1)      # out: [None, n_heads, n_actions]
    z_var     = self._compute_z_variance(agent_net)           # out: [None, n_actions]
    self.rho2 = z_var + 1e-5
    # self.rho2   = 0.5**2    # Const for IDS Info Gain

    # Compute the IDS action - ugly way
    action    = BstrapDQN_IDS._act_train(self, q_heads, name)

    # Add debugging data for TB
    tf.summary.histogram("debug/a_rho2", self.rho2)
    tf.summary.scalar("debug/z_var", tf.reduce_mean(z_var))

    p_rho2  = tf.identity(self.rho2[0], name="plot/train/rho2")
    p_a     = self.plot_train["train_actions"]["a_mean"]["a"]

    self.plot_train["train_actions"]["a_rho2"] = dict(height=p_rho2,  a=p_a)

    return action


  def _act_eval(self, agent_net, name):
    bins  = tf.expand_dims(self.bins, axis=0)
    q     = tf.reduce_sum(agent_net * bins, axis=-1)      # out: [None, n_heads, n_actions]
    return self._act_eval_greedy(q, name)
    # return self._act_eval_vote(q, name)
