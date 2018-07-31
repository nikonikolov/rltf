import numpy      as np
import tensorflow as tf

from rltf.models.dqn  import BaseDQN


class C51(BaseDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, V_min, V_max, N):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
      V_min: float. lower bound for histrogram range
      V_max: float. upper bound for histrogram range
      N: int. number of histogram bins
    """

    super().__init__(obs_shape, n_actions, opt_conf, gamma)

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
    """ Build the C51 architecture - as desribed in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created
    Returns:
      `tf.Tensor` of shape `[batch_size, n_actions, N]`. Contains the distribution of Q for each action
    """
    n_actions = self.n_actions
    N         = self.N

    with tf.variable_scope("conv_net"):
      # original architecture
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    with tf.variable_scope("action_value"):
      x = tf.layers.dense(x, units=512,          activation=tf.nn.relu)
      x = tf.layers.dense(x, units=N*n_actions,  activation=None)

    # Compute Softmax probabilities in numerically stable way
    x = tf.reshape(x, [-1, n_actions, N])
    C = tf.stop_gradient(tf.reduce_max(x, axis=-1, keepdims=True))
    x = tf.nn.softmax(x-C, axis=-1)
    return x


  def _compute_estimate(self, agent_net):
    """Select the return distribution Z of the selected action
    Args:
      agent_net: `tf.Tensor`, shape `[None, n_actions, N]. The tensor output from `self._nn_model()`
        for the agent
    Returns:
      `tf.Tensor` of shape `[None, N]`
    """
    a_mask  = tf.one_hot(self._act_t_ph, self.n_actions, dtype=tf.float32)  # out: [None, n_actions]
    a_mask  = tf.expand_dims(a_mask, axis=-1)                               # out: [None, n_actions, 1]
    z       = tf.reduce_sum(agent_net * a_mask, axis=1)                     # out: [None, N]
    return z


  def _select_target(self, target_net):
    """Select the C51 target distributions - use the greedy action from E[Z]
    Args:
      target_net: `tf.Tensor`, shape `[None, n_actions, N]. The tensor output from `self._nn_model()`
        for the target
    Returns:
      `tf.Tensor` of shape `[None, N]`
    """
    n_actions   = self.n_actions
    target_z    = target_net

    # Get the target Q probabilities for the greedy action; output shape [None, N]
    target_q    = tf.reduce_sum(target_z * self.bins, axis=-1)            # out: [None, n_actions]
    target_act  = tf.argmax(target_q, axis=-1, output_type=tf.int32)      # out: [None]
    target_mask = tf.one_hot(target_act, n_actions, dtype=tf.float32)     # out: [None, n_actions]
    target_mask = tf.expand_dims(target_mask, axis=-1)                    # out: [None, n_actions, 1]
    target_z    = tf.reduce_sum(target_z * target_mask, axis=1)           # out: [None, N]
    return target_z


  def _compute_backup(self, target):
    """Compute the C51 backup distributions
    Args:
      target: `tf.Tensor`, shape `[None, N]. The output from `self._select_target()`
    Returns:
      `tf.Tensor` of shape `[None, N]`
    """

    def build_inds_tensors(bin_inds_lo, bin_inds_hi):
      batch       = tf.shape(self.done_ph)[0]

      batch_inds  = tf.range(0, limit=batch, delta=1, dtype=tf.int32)
      batch_inds  = tf.expand_dims(batch_inds, axis=-1)               # out: [None, 1]
      batch_inds  = tf.tile(batch_inds, [1, self.N])                  # out: [None, N]
      batch_inds  = tf.expand_dims(batch_inds, axis=-1)               # out: [None, N, 1]

      bin_inds_lo = tf.expand_dims(tf.to_int32(bin_inds_lo), axis=-1) # out: [None, N, 1]
      bin_inds_hi = tf.expand_dims(tf.to_int32(bin_inds_hi), axis=-1) # out: [None, N, 1]
      bin_inds_lo = tf.concat([batch_inds, bin_inds_lo], axis=-1)     # out: [None, N, 2]
      bin_inds_hi = tf.concat([batch_inds, bin_inds_hi], axis=-1)     # out: [None, N, 2]

      return bin_inds_lo, bin_inds_hi

    target_z    = target

    # Compute projected bin support; output shape [None, N]
    done_mask   = tf.cast(tf.logical_not(self.done_ph), tf.float32)
    done_mask   = tf.expand_dims(done_mask, axis=-1)
    rew_t       = tf.expand_dims(self.rew_t_ph, axis=-1)
    bins        = tf.squeeze(self.bins, axis=0)
    target_bins = rew_t + self.gamma * done_mask * bins
    target_bins = tf.clip_by_value(target_bins, self.V_min, self.V_max)

    # Projected bin indices; output shape [None, N], dtype=float
    bin_inds    = (target_bins - self.V_min) / self.dz
    bin_inds_lo = tf.floor(bin_inds)
    bin_inds_hi = tf.ceil(bin_inds)

    lo_add      = target_z * (bin_inds_hi - bin_inds)
    hi_add      = target_z * (bin_inds - bin_inds_lo)

    # Initialize the Variable holding the target distribution - gets reset to 0 every time
    zeros       = tf.zeros_like(target_bins, dtype=tf.float32)
    target_z    = tf.Variable(0, trainable=False, dtype=tf.float32, validate_shape=False)
    target_z    = tf.assign(target_z, zeros, validate_shape=False)

    # Compute indices for scatter_nd_add
    inds        = build_inds_tensors(bin_inds_lo, bin_inds_hi)
    bin_inds_lo = inds[0]     # out: [None, N, 2]
    bin_inds_hi = inds[1]     # out: [None, N, 2]

    with tf.control_dependencies([target_z]):
      target_z  = tf.scatter_nd_add(target_z, bin_inds_lo, lo_add, use_locking=True)
      target_z  = tf.scatter_nd_add(target_z, bin_inds_hi, hi_add, use_locking=True)

    return target_z


  def _compute_loss(self, estimate, target, name):
    z         = estimate
    target_z  = target
    entropy   = -tf.reduce_sum(target_z * tf.log(z), axis=-1)
    loss      = tf.reduce_mean(entropy)

    tf.summary.scalar(name, loss)

    return loss


  def _act_train(self, agent_net, name):
    # Compute the Q-function as expectation of Z; output shape [None, n_actions]
    q       = tf.reduce_sum(agent_net * self.bins, axis=-1)
    action  = tf.argmax(q, axis=-1, output_type=tf.int32, name=name)

    # Add debugging plot for the variance of the return
    center  = self.bins - tf.expand_dims(q, axis=-1)      # out: [None, n_actions, N]
    z_var   = tf.square(center) * agent_net               # out: [None, n_actions, N]
    z_var   = tf.reduce_sum(z_var, axis=-1)               # out: [None, n_actions]
    tf.summary.scalar("debug/z_var", tf.reduce_mean(z_var))
    tf.summary.histogram("debug/a_rho2", z_var)

    return action


  def _act_eval(self, agent_net, name):
    return tf.identity(self.a_train, name=name)



class C51TS(C51):

  def _act_train(self, agent_net, name):
    """
    Args:
      agent_net: tf.Tensor, shape [None, n_actions, N]
    """

    # samples; out shape: [None, n_actions, N]
    sample  = tf.random_uniform([tf.shape(agent_net)[0], 1], 0.0, 1.0)  # out shape: [None, 1]
    sample  = tf.tile(sample, [1, self.n_actions])                      # out shape: [None, n_actions]
    sample  = tf.expand_dims(sample, axis=-1)                           # out shape: [None, n_actions, N]

    # CDF of each action; out shape: [None, n_actions, N]
    cdf     = tf.cumsum(agent_net, axis=-1, exclusive=True)

    # Find the sampled bin indices; out shape: [None, n_actions]
    offset  = tf.where(cdf <= sample, tf.zeros_like(cdf), -2*tf.ones_like(cdf))
    inds    = tf.argmax(cdf + offset, axis=-1, output_type=tf.int32)

    # Get the value of the sampled bin; out shape: [None, n_actions]
    inds    = tf.one_hot(inds, self.N, axis=-1, dtype=tf.float32)       # out shape: [None, n_actions, N]
    q       = tf.reduce_sum(self.bins * inds, axis=-1)                  # out shape: [None, n_actions]

    action  = tf.argmax(q, axis=-1, output_type=tf.int32, name=name)    # out shape: [None]

    return action


  def _select_target(self, target_net):
    # Compute the Double DQN target
    target_z    = target_net                                                # out: [None, n_actions, N]

    # Compute the Z and Q estimates with the agent network variables
    agent_z     = self._nn_model(self._obs_tp1, scope="agent_net")          # out: [None, n_actions, N]
    agent_q     = tf.reduce_mean(agent_z * self.bins, axis=-1)              # out: [None, n_actions]

    # Get the target Q probabilities for the greedy action
    target_act  = tf.argmax(agent_q, axis=-1)                               # out: [None]
    target_mask = tf.one_hot(target_act, self.n_actions, dtype=tf.float32)  # out: [None, n_actions]
    target_mask = tf.expand_dims(target_mask, axis=-1)                      # out: [None, n_actions, 1]
    target_z    = tf.reduce_sum(target_z * target_mask, axis=1)             # out: [None, N]
    return target_z


  def _act_eval(self, agent_net, name):
    # Compute the Q-function as expectation of Z; output shape [None, n_actions]
    q       = tf.reduce_sum(agent_net * self.bins, axis=-1)
    action  = tf.argmax(q, axis=-1, output_type=tf.int32, name=name)
    return action
