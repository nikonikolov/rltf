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
