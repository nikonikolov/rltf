import tensorflow as tf

from rltf.models import BaseDQN


class DQN(BaseDQN):

  def __init__(self, huber_loss, **kwargs):
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


  def _conv_nn(self, x):
    """ Build the DQN architecture - as described in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created
    Returns:
      `tf.Tensor` of shape `[batch_size, n_actions]`. Contains the Q-function for each action
    """
    n_actions = self.n_actions

    with tf.variable_scope("conv_net"):
      # original architecture
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    with tf.variable_scope("action_value"):
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
      x = tf.layers.dense(x, units=n_actions, activation=None)
    return x


  def _dense_nn(self, x):
    """ Build a Neural Network of dense layers only. Used for low-level observations
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created
    Returns:
      `tf.Tensor` of shape `[batch_size, n_actions]`. Contains the Q-function for each action
    """
    n_actions = self.n_actions

    with tf.variable_scope("dense_net"):
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
      x = tf.layers.dense(x, units=n_actions, activation=None)
    return x


  def _act_train(self, agent_net, name):
    action = tf.argmax(agent_net, axis=-1, output_type=tf.int32, name=name)
    return dict(action=action)


  def _act_eval(self, agent_net, name):
    return dict(action=tf.identity(self.train_dict["action"], name=name))


  def _compute_estimate(self, agent_net):
    # Get the Q value for the selected action; output shape [None]
    a_mask  = tf.one_hot(self.act_t_ph, self.n_actions, dtype=tf.float32)
    q       = tf.reduce_sum(agent_net * a_mask, axis=-1)

    return q


  def _select_target(self, target_net):
    target_q  = tf.reduce_max(target_net, axis=-1)
    return target_q


  def _compute_backup(self, target):
    done_mask = tf.cast(tf.logical_not(self.done_ph), tf.float32)
    target_q  = self.rew_t_ph + self.gamma * done_mask * target
    return target_q


  def _compute_loss(self, estimate, target, name):
    if self.huber_loss:
      loss = tf.losses.huber_loss(target, estimate)
    else:
      loss = tf.losses.mean_squared_error(target, estimate)
    tf.summary.scalar(name, loss)
    return loss
