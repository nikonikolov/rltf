import numpy      as np
import tensorflow as tf

from rltf.models.dqn  import BaseDQN
from rltf.models      import tf_utils


class QRDQN(BaseDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, N, k):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      N: int. number of quantiles
      k: int. Huber loss order
    """

    super().__init__(obs_shape, n_actions, opt_conf, gamma)

    self.N = N
    self.k = k


  def _conv_nn(self, x):
    """ Build the QR DQN architecture - as desribed in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created

    Returns:
      `tf.Tensor` of shape `[batch_size, n_actions, N]`. Contains the distribution of Q for each action
    """
    n_actions = self.n_actions
    N         = self.N
    init_glorot_normal = tf_utils.init_glorot_normal

    with tf.variable_scope("conv_net"):
      # original architecture
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=init_glorot_normal())
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=init_glorot_normal())
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=init_glorot_normal())
    x = tf.layers.flatten(x)
    with tf.variable_scope("action_value"):
      x = tf.layers.dense(x, 512,         activation=tf.nn.relu,  kernel_initializer=init_glorot_normal())
      x = tf.layers.dense(x, N*n_actions, activation=None,        kernel_initializer=init_glorot_normal())

    x = tf.reshape(x, [-1, n_actions, N])
    return x


  def _compute_estimate(self, agent_net):
    # Get the Z-distribution for the selected action; output shape [None, N]
    a_mask  = tf.expand_dims(tf.one_hot(self._act_t_ph, self.n_actions, dtype=tf.float32), axis=-1)
    z       = tf.reduce_sum(agent_net * a_mask, axis=1)
    return z


  def _compute_target(self, target_net):
    target_z      = target_net

    # Compute the Q-function as expectation of Z; output shape [None, n_actions]
    target_q      = tf.reduce_mean(target_z, axis=-1)

    # Get the target Q probabilities for the greedy action; output shape [None, N]
    target_act    = tf.argmax(target_q, axis=-1)
    a_mask        = tf.expand_dims(tf.one_hot(target_act, self.n_actions, dtype=tf.float32), axis=-1)
    target_z      = tf.reduce_sum(target_z * a_mask, axis=1)

    # Compute the projected quantiles; output shape [None, N]
    done_mask     = tf.cast(tf.logical_not(self.done_ph), tf.float32)
    done_mask     = tf.expand_dims(done_mask, axis=-1)
    rew_t         = tf.expand_dims(self.rew_t_ph, axis=-1)
    target_z      = rew_t + self.gamma * done_mask * target_z
    target_z      = tf.stop_gradient(target_z)

    return target_z


  def _compute_loss(self, estimate, target):
    z             = estimate
    target_z      = target

    # Compute the tensor of mid-quantiles
    mid_quantiles = (np.arange(0, self.N, 1, dtype=np.float64) + 0.5) / float(self.N)
    mid_quantiles = np.asarray(mid_quantiles, dtype=np.float32)
    mid_quantiles = tf.constant(mid_quantiles[None, None, :], dtype=tf.float32)

    # Operate over last dimensions to get result for for theta_i
    z_diff        = tf.expand_dims(target_z, axis=-2) - tf.expand_dims(z, axis=-1)
    indicator_fn  = tf.to_float(z_diff < 0.0)

    penalty_w     = mid_quantiles - indicator_fn

    # Pure Quantile Regression Loss
    if self.k == 0:
      huber_loss  = z_diff
    # Quantile Huber Loss
    else:
      penalty_w   = tf.abs(penalty_w)
      huber_loss  = tf_utils.huber_loss(z_diff, delta=np.float32(self.k))

    quantile_loss = huber_loss * penalty_w
    quantile_loss = tf.reduce_mean(quantile_loss, axis=-1)
    loss          = tf.reduce_sum(quantile_loss, axis=-1)
    loss          = tf.reduce_mean(loss)

    tf.summary.scalar("train/loss", loss)

    return loss


  def _act_train(self, agent_net, name):
    # Compute the Q-function as expectation of Z; output shape [None, n_actions]
    q       = tf.reduce_mean(agent_net, axis=-1)
    action  = tf.argmax(q, axis=-1, output_type=tf.int32, name=name)
    return action


  def _act_eval(self, agent_net, name):
    return tf.identity(self.a_train, name=name)


class QRDQNTS(QRDQN):

  def _act_train(self, agent_net, name):

    # samples; out shape: [n_actions, 1]
    sample = tf.random_uniform([self.n_actions, 1], 0.0, 1.0)

    # quantiles; out shape [n_actions, N]
    quantiles = np.arange(0, self.N, 1, dtype=np.float32) / float(self.N-1)
    quantiles = tf.constant(quantiles[None, :], dtype=tf.float32)
    quantiles = tf.tile(quantiles, [self.n_actions, 1])

    zeros     = tf.zeros_like(quantiles)
    offset    = tf.ones_like(quantiles) * self.N
    
    # Find the lower quantile; out shape [n_actions, 1]
    lo_offset = tf.where(tf.greater(quantiles - sample, 0), -offset, zeros)
    lo_inds   = tf.argmax(quantiles + lo_offset, axis=-1, output_type=tf.int32)
    lo_inds   = tf.expand_dims(lo_inds, axis=-1)

    # Find the higher quantile; out shape [n_actions, 1]
    hi_offset = tf.where(tf.less(quantiles - sample, 0), offset, zeros)
    hi_inds   = tf.argmin(quantiles + hi_offset, axis=-1, output_type=tf.int32)
    hi_inds   = tf.expand_dims(hi_inds, axis=-1)
    
    # Batch indices; out shape: [n_actions, 1]
    b_inds = tf.zeros([self.n_actions, 1], dtype=tf.int32)

    # Action indices; out shape: [n_actions, 1]
    a_inds = tf.range(0, limit=self.n_actions, delta=1, dtype=tf.int32)
    a_inds = tf.expand_dims(a_inds, axis=-1)

    # lo and hi Q indices; out shape: [n_actions, 3]
    g_lo_inds = tf.concat([b_inds, a_inds, lo_inds], axis=-1)
    g_hi_inds = tf.concat([b_inds, a_inds, hi_inds], axis=-1)

    # Q lo and hi; out shape: [n_actions, 1]
    lo = tf.gather_nd(agent_net, g_lo_inds)
    hi = tf.gather_nd(agent_net, g_hi_inds)
    hi = tf.expand_dims(hi, axis=-1)
    lo = tf.expand_dims(lo, axis=-1)

    # Interpolated Q; out shape [n_actions, 1]
    alpha     = (hi - lo) / float(self.N-1)
    lo_quant  = tf.cast(lo_inds, tf.float32) * float(self.N-1)
    q = lo + alpha * (sample - lo_quant)

    # Get the greedy action; out shape [1]
    action = tf.argmax(q, axis=0, name=name)
    
    return action


  def _compute_target(self, target_net):
    target_z    = target_net

    # Compute the Z and Q estimate swith the agent network variables
    agent_z     = self._nn_model(self._obs_tp1, scope="agent_net")
    agent_q     = tf.reduce_mean(agent_z, axis=-1)

    # Get the target Q probabilities for the greedy action; output shape [None, N]
    target_act  = tf.argmax(agent_q, axis=-1)
    a_mask      = tf.expand_dims(tf.one_hot(target_act, self.n_actions, dtype=tf.float32), axis=-1)
    target_z    = tf.reduce_sum(target_z * a_mask, axis=1)

    # Compute the projected quantiles; output shape [None, N]
    done_mask   = tf.cast(tf.logical_not(self.done_ph), tf.float32)
    done_mask   = tf.expand_dims(done_mask, axis=-1)
    rew_t       = tf.expand_dims(self.rew_t_ph, axis=-1)
    target_z    = rew_t + self.gamma * done_mask * target_z
    target_z    = tf.stop_gradient(target_z)

    return target_z


  def _act_eval(self, agent_net, name):
    # Compute the Q-function as expectation of Z; output shape [None, n_actions]
    q       = tf.reduce_mean(agent_net, axis=-1)
    action  = tf.argmax(q, axis=-1, output_type=tf.int32, name=name)
    return action
