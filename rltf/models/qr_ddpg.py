import logging
import numpy      as np
import tensorflow as tf

from rltf.models  import DDPG
from rltf.models  import tf_utils

from rltf.models.ddpg import init_hidden_uniform

logger = logging.getLogger(__name__)


class QRDDPG(DDPG):

  def __init__(self, obs_shape, n_actions, actor_opt_conf, critic_opt_conf,
               critic_reg, tau, gamma, N, huber_loss):

    super().__init__(obs_shape, n_actions, actor_opt_conf, critic_opt_conf,
                     critic_reg, tau, gamma, huber_loss)
    self.N = N
    # self.hidden_init = tf_utils.init_default
    # self.output_init = tf_utils.init_default

    # Custom TF Tensors and Ops
    self.quantile_mids = None


  def build(self):
    quantile_mids       = (np.arange(0, self.N, 1, dtype=np.float64) + 0.5) / float(self.N)
    quantile_mids       = np.asarray(quantile_mids, dtype=np.float32)
    self.quantile_mids  = tf.constant(quantile_mids[None, None, :], dtype=tf.float32)

    super().build()


  # def _compute_target(self, target_q):
  #   done_mask = tf.cast(tf.logical_not(self._done_ph), tf.float32)
  #   done_mask = tf.expand_dims(done_mask, axis=-1)
  #   rew_t_ph  = tf.expand_dims(self.rew_t_ph, axis=-1)
  #   return rew_t_ph + done_mask * self.gamma * target_q


  def _get_actor_loss(self, actor_critic_q):
    actor_loss = tf.reduce_mean(actor_critic_q, axis=-1)
    actor_loss = -tf.reduce_mean(actor_loss)
    return actor_loss


  def _get_critic_loss(self, target_q, agent_q):
    """
    Args:
      target_q: tf.Tensor. The output of self._compute_target()
      agent_q: tf.Tensor. The output of the self._critic_net() with actions
        sampled from the buffer as input
    """

    # Operate over last dimensions to get result for for theta_i
    quantile_diffs  = tf.expand_dims(target_q, axis=-2) - tf.expand_dims(agent_q, axis=-1)
    indicator       = tf.to_float(quantile_diffs < 0.0)

    quants_loss     = self.quantile_mids - indicator

    # Quantile Huber Loss
    if self.huber_loss == 0:
      quants_loss   = tf.abs(quants_loss)
      huber_loss    = tf_utils.huber_loss(quantile_diffs, delta=1.0)
    # Quantile Regression Loss
    else:
      huber_loss    = quantile_diffs

    samples_loss    = huber_loss * quants_loss
    expected_loss   = tf.reduce_mean(samples_loss, axis=-1)
    quantile_loss   = tf.reduce_sum(expected_loss, axis=-1)
    loss            = tf.reduce_mean(quantile_loss)
    critic_loss     = loss + tf.losses.get_regularization_loss(scope="agent_net/critic")

    return critic_loss


  # def _critic_net(self, state, action, scope):
  #   """Build critic network

  #   Args:
  #     state: tf.Tensor. Input tensor for the state. Batch must be the 0 dimension
  #     action: tf.Tensor. Input tensor for the action. Batch must be the 0 dimension
  #     scope: string. Parent scope for the network variables. Must end in "/"
  #   Returns:
  #     `tf.Tensor` that holds the value of the Q-function estimate
  #   """

  #   regularizer = tf.contrib.layers.l2_regularizer(scale=self.critic_reg)
  #   x = state
  #   with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
  #     x = tf.layers.dense(x, 400, tf.nn.relu, kernel_initializer=self.hidden_init(),
  #                         kernel_regularizer=regularizer, name="dense1")
  #     x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")

  #     x = tf.concat([x, action], axis=-1)

  #     # No batch norm after action input, as in the original paper
  #     x = tf.layers.dense(x, 300, tf.nn.relu, kernel_initializer=self.hidden_init(),
  #                         kernel_regularizer=regularizer, name="dense2")

  #     x = tf.layers.dense(x, self.N, kernel_initializer=self.output_init(),
  #                         kernel_regularizer=regularizer, name="dense3")
  #     return x


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
      x = tf.layers.dense(x, 400, kernel_initializer=self.hidden_init(),
                          kernel_regularizer=regularizer, name="dense1")
      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm1")
      x = tf.nn.relu(x)

      x = tf.concat([x, action], axis=-1)

      # No batch norm after action input, as in the original paper
      x = tf.layers.dense(x, 300, kernel_initializer=self.hidden_init(),
                          kernel_regularizer=regularizer, name="dense2")

      x = tf.layers.batch_normalization(x, axis=-1, training=self._training, name="batch_norm2")
      x = tf.nn.relu(x)

      x = tf.layers.dense(x, self.N, kernel_initializer=self.output_init(),
                          kernel_regularizer=regularizer, name="dense3")
      return x
