import tensorflow as tf

from rltf.models  import BaseDQN


class BstrapDQN(BaseDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, huber_loss, n_heads):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
      huber_loss: bool. Whether to use huber loss or not
    """

    super().__init__(obs_shape, n_actions, opt_conf, gamma)

    self.huber_loss   = huber_loss
    self.n_heads      = n_heads

    # Custom TF Tensors and Ops
    self.active_head  = None
    self.set_act_head = None


  def build(self):
    self.active_head  = tf.Variable([0], trainable=False, name="active_head")
    sample_head       = tf.random_uniform(shape=[1], maxval=self.n_heads, dtype=tf.int32)
    self.set_act_head = tf.assign(self.active_head, sample_head, name="set_act_head")

    super().build()


  def _nn_model(self, x, scope):
    """ Build the DQN architecture - as described in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created
    Returns:
      `tf.Tensor` of shape `[batch_size, n_heads, n_actions]`. Contains the Q-function for each action
    """

    def _build_head(x, n_actions):
      """ Build the head of the DQN network
      Args:
        x: tf.Tensor. Tensor for the input
      Returns:
        `tf.Tensor` of shape `[batch_size, n_actions]`. Contains the Q-function for each action
      """
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
      x = tf.layers.dense(x, units=n_actions, activation=None)
      x = tf.expand_dims(x, axis=-2)
      return x

    n_actions = self.n_actions
    with tf.variable_scope(scope, reuse=False):
      with tf.variable_scope("convnet"):
        x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.flatten(x)
      with tf.variable_scope("action_value"):
        heads = [_build_head(x, n_actions) for _ in range(self.n_heads)]
        x = tf.concat(heads, axis=-2)
      return x


  def _compute_q(self, nn_out):
    head_mask = tf.one_hot(self.active_head, self.n_heads, on_value=True, off_value=False, dtype=tf.bool)
    q_head    = tf.boolean_mask(nn_out, head_mask)
    return q_head


  def _compute_estimate(self, nn_out):
    """Get the Q value for the selected action
    Returns:
      `tf.Tensor` of shape `[None, n_heads]`
    """
    q         = nn_out
    act_t     = tf.cast(self._act_t_ph, tf.int32)
    act_mask  = tf.one_hot(act_t, self.n_actions, on_value=True, off_value=False, dtype=tf.bool)
    act_mask  = tf.expand_dims(act_mask, axis=-2)
    act_mask  = tf.tile(act_mask, [1, self.n_heads, 1])
    q         = tf.boolean_mask(q, act_mask)
    q         = tf.reshape(q, shape=[-1, self.n_heads])
    return q


  def _compute_target(self, nn_out):
    """Compute the backup value of the greedy action
    Returns:
      `tf.Tensor` of shape `[None, n_heads]`
    """
    target_q  = nn_out
    done_mask = tf.cast(tf.logical_not(self._done_ph), tf.float32)
    target_q  = tf.reduce_max(target_q, axis=-1)
    done_mask = tf.expand_dims(done_mask, axis=-1)
    rew_t     = tf.expand_dims(self.rew_t_ph, axis=-1)
    target_q  = rew_t + self.gamma * done_mask * target_q

    return target_q


  def _compute_loss(self, estimate, target):
    loss_fn   = tf.losses.huber_loss if self.huber_loss else tf.losses.mean_squared_error
    loss      = loss_fn(target, estimate)

    return loss


  def reset(self, sess):
    sess.run(self.set_act_head)
