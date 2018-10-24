import numpy      as np
import tensorflow as tf

from rltf.models.dqn  import BaseDQN
from rltf.models      import tf_utils


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
    # C = tf.stop_gradient(tf.reduce_max(x, axis=-1, keepdims=True))
    # x = tf.nn.softmax(x-C, axis=-1)
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
    target_z    = tf_utils.softmax(target_net, axis=-1)

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
    # bins        = tf.squeeze(self.bins, axis=0)
    bins        = tf.reshape(self.bins, [1, self.N])
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
    logits_z  = estimate
    target_z  = target
    # Working version
    entropy   = -tf.reduce_sum(target_z * tf.log(tf_utils.softmax(logits_z, axis=-1)), axis=-1)
    # Other version - not working
    # entropy   = -tf.reduce_sum(target_z * tf_utils.log_softmax(logits_z, axis=-1), axis=-1)
    # Other version - not learning
    # entropy   = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_z, logits=logits_z)
    loss      = tf.reduce_mean(entropy)

    tf.summary.scalar(name, loss)

    return loss


  def _act_train(self, agent_net, name):
    # Compute the Q-function as expectation of Z; output shape [None, n_actions]
    z       = tf_utils.softmax(agent_net, axis=-1)
    q       = tf.reduce_sum(z * self.bins, axis=-1)
    action  = tf.argmax(q, axis=-1, output_type=tf.int32, name=name)

    # Add debugging plot for the variance of the return
    z_var   = self._compute_z_variance(z=z, q=q, normalize=True)  # [None, n_actions]
    tf.summary.scalar("debug/z_var", tf.reduce_mean(z_var))
    tf.summary.histogram("debug/a_rho2", z_var)

    return action


  def _act_eval(self, agent_net, name):
    return tf.identity(self.a_train, name=name)


  def _compute_z_variance(self, z=None, logits=None, q=None, normalize=True):
    """Compute the return distribution variance. Only one of `z` and `logits` must be set
    Args:
      z: tf.Tensor, shape `[None, n_actions, N]`. Return atoms probabilities
      logits: tf.Tensor, shape `[None, n_actions, N]`. Logits of the return
      q: tf.Tensor, shape `[None, n_actions]`. Optionally provide a tensor for the Q-function
      normalize: bool. If True, normalize the variance values such that the mean of the
        return variances of all actions in a given state is 1.
    Returns:
      tf.Tensor of shape `[None, n_actions]`
    """
    assert (z is None) != (logits is None), "Only one of 'z' and 'logits' must be set"

    if logits is not None:
      z = tf_utils.softmax(logits, axis=-1)
    if q is None:
      q = tf.reduce_sum(z * self.bins, axis=-1)

    center  = self.bins - tf.expand_dims(q, axis=-1)          # out: [None, n_actions, N]
    z_var   = tf.square(center) * z                           # out: [None, n_actions, N]
    z_var   = tf.reduce_sum(z_var, axis=-1)                   # out: [None, n_actions]

    # Normalize the variance across the action axis
    if normalize:
      mean  = tf.reduce_mean(z_var, axis=-1, keepdims=True)   # out: [None, 1]
      z_var = z_var / (mean + 1e-6)                           # out: [None, n_actions]

    return z_var
