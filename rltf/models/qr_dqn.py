import numpy      as np
import tensorflow as tf

from rltf.models  import DQN
from rltf.models  import tf_utils


def qr_dqn_nn(x, n_actions, N, scope):
  """ Build the QR DQN architecture - as desribed in the original paper

  Args:
    x: tf.Tensor. Tensor for the input
    n_actions: int. Number of possible output actions
    N: int. Number of quantiles
    scope: str. Scope in which all the model related variables should be created

  Returns:
    `tf.Tensor` of shape `[batch_size, n_actions, N]`. Contains the distribution of Q for each action
  """
  init_glorot_normal = tf_utils.init_glorot_normal

  with tf.variable_scope(scope, reuse=False):
    with tf.variable_scope("convnet"):
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


class QRDQN(DQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, N, k):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      N: int. number of quantiles
      k: int. Huber loss order
    """

    huber_loss = True if k == 1 else False
    super().__init__(obs_shape, n_actions, opt_conf, gamma, huber_loss)

    self.N = N
    self.k = k


  def build(self):

    super()._build()

    # In this case, casting on GPU ensures lower data transfer times
    obs_t_float   = tf.cast(self._obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(self._obs_tp1_ph, tf.float32) / 255.0
    act_t         = tf.cast(self._act_t_ph,   tf.int32)

    # Compute the tensor of mid-quantiles
    mid_quantiles = (np.arange(0, self.N, 1, dtype=np.float64) + 0.5) / float(self.N)
    mid_quantiles = np.asarray(mid_quantiles, dtype=np.float32)
    mid_quantiles = tf.constant(mid_quantiles[None, None, :], dtype=tf.float32)

    # Construct the Z-network and the target network; output shape [None, n_actions, N]
    z             = qr_dqn_nn(obs_t_float,   n_actions=self.n_actions, N=self.N, scope="agent_net")
    target_z      = qr_dqn_nn(obs_tp1_float, n_actions=self.n_actions, N=self.N, scope="target_net")

    # Compute the Q-function as expectation of Z; output shape [None, n_actions]
    q             = tf.reduce_mean(z,         axis=-1)
    target_q      = tf.reduce_mean(target_z,  axis=-1)

    # Get the Z-distribution for the selected action; output shape [None, N]
    # batch         = tf.shape(self.done_ph)[0]
    # inds          = tf.range(0, limit=batch, delta=1, dtype=tf.int32)
    # inds          = tf.concat([inds, act_t], axis=-1)
    # z             = tf.gather_nd(z, inds)
    act_mask      = tf.one_hot(act_t, self.n_actions, on_value=True, off_value=False, dtype=tf.bool)
    z             = tf.boolean_mask(z, act_mask)

    # Get the target Q probabilities for the greedy action; output shape [None, N]
    target_act    = tf.argmax(target_q, axis=-1)
    t_act_mask    = tf.one_hot(target_act, self.n_actions, on_value=True, off_value=False, dtype=tf.bool)
    target_z      = tf.boolean_mask(target_z, t_act_mask)

    # Compute the projected quantiles; output shape [None, N]
    done_mask     = tf.cast(tf.logical_not(self.done_ph), tf.float32)
    done_mask     = tf.expand_dims(done_mask, axis=-1)
    rew_t         = tf.expand_dims(self.rew_t_ph, axis=-1)
    target_z      = rew_t + self.gamma * done_mask * target_z


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

    agent_vars    = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_net')
    target_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

    # Build the optimizer
    optimizer   = self.opt_conf.build()
    # Create the training Op
    self._train_op = optimizer.minimize(loss, var_list=agent_vars, name="train_op")

    # Create the Op to update the target
    self._update_target = tf_utils.assign_vars(target_vars, agent_vars, name="update_target")

    # Save the Q-function estimate tensor
    self._q   = tf.identity(q, name="q_fn")

    # Add summaries
    tf.summary.scalar("loss", loss)
