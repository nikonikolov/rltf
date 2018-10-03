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
    # init      = tf_utils.init_dqn
    init      = tf_utils.init_glorot_normal
    # init      = tf_utils.init_default

    with tf.variable_scope("conv_net"):
      # original architecture
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=init(), bias_initializer=init())
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=init(), bias_initializer=init())
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                           kernel_initializer=init(), bias_initializer=init())
    x = tf.layers.flatten(x)
    with tf.variable_scope("action_value"):
      x = tf.layers.dense(x, 512,         activation=tf.nn.relu,
                           kernel_initializer=init(), bias_initializer=init())
      x = tf.layers.dense(x, N*n_actions, activation=None,
                           kernel_initializer=init(), bias_initializer=init())

    x = tf.reshape(x, [-1, n_actions, N])
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
    """Select the QRDQN target distributions - use the greedy action from E[Z]
    Args:
      target_net: `tf.Tensor`, shape `[None, n_actions, N]. The tensor output from `self._nn_model()`
        for the target
    Returns:
      `tf.Tensor` of shape `[None, N]`
    """
    # Compute the Q-function as expectation of Z
    target_z    = target_net                                                # out: [None, n_actions, N]
    target_q    = tf.reduce_mean(target_z, axis=-1)                         # out: [None, n_actions]

    # Get the target Q probabilities for the greedy action
    target_act  = tf.argmax(target_q, axis=-1)                              # out: [None]
    target_mask = tf.one_hot(target_act, self.n_actions, dtype=tf.float32)  # out: [None, n_actions]
    target_mask = tf.expand_dims(target_mask, axis=-1)                      # out: [None, n_actions, 1]
    target_z    = tf.reduce_sum(target_z * target_mask, axis=1)             # out: [None, N]
    return target_z


  def _compute_backup(self, target):
    """Compute the QRDQN backup distributions
    Args:
      target: `tf.Tensor`, shape `[None, N]. The output from `self._select_target()`
    Returns:
      `tf.Tensor` of shape `[None, N]`
    """
    # Compute the projected quantiles; output shape [None, N]
    target_z  = target
    done_mask = tf.cast(tf.logical_not(self.done_ph), tf.float32)
    done_mask = tf.expand_dims(done_mask, axis=-1)
    rew_t     = tf.expand_dims(self.rew_t_ph, axis=-1)
    target_z  = rew_t + self.gamma * done_mask * target_z
    return target_z


  def _compute_loss(self, estimate, target, name):
    """Compute the QRDQN loss.
    Args:
      agent_net: `tf.Tensor`, shape `[None, N]. The tensor output from `self._compute_estimate()`
      target_net: `tf.Tensor`, shape `[None, N]. The tensor output from `self._compute_target()`
    Returns:
      `tf.Tensor` of scalar shape `()`
    """
    z             = estimate
    target_z      = target

    # Compute the tensor of mid-quantiles
    mid_quantiles = (np.arange(0, self.N, 1, dtype=np.float64) + 0.5) / float(self.N)
    mid_quantiles = np.asarray(mid_quantiles, dtype=np.float32)
    mid_quantiles = tf.constant(mid_quantiles[None, None, :], dtype=tf.float32)

    # Operate over last dimensions to average over samples (target locations)
    td_z          = tf.expand_dims(target_z, axis=-2) - tf.expand_dims(z, axis=-1)
    # td_z[0] =
    # [ [tz1-z1, tz2-z1, ..., tzN-z1],
    #   [tz1-z2, tz2-z2, ..., tzN-z2],
    #   ...
    #   [tz1-zN, tzN-zN, ..., tzN-zN]  ]
    indicator_fn  = tf.to_float(td_z < 0.0)                   # out: [None, N, N]

    # Compute the quantile penalty weights
    quant_weight  = mid_quantiles - indicator_fn              # out: [None, N, N]
    # Make sure no gradient flows through the indicator function. The penalty is only a scaling factor
    quant_weight  = tf.stop_gradient(quant_weight)

    # Pure Quantile Regression Loss
    if self.k == 0:
      quantile_loss = quant_weight * td_z                     # out: [None, N, N]
    # Quantile Huber Loss
    else:
      quant_weight  = tf.abs(quant_weight)
      huber_loss    = tf_utils.huber_loss(td_z, delta=np.float32(self.k))
      quantile_loss = quant_weight * huber_loss               # out: [None, N, N]

    quantile_loss = tf.reduce_mean(quantile_loss, axis=-1)    # Expected loss for each quntile
    loss          = tf.reduce_sum(quantile_loss, axis=-1)     # Sum loss over all quantiles
    loss          = tf.reduce_mean(loss)                      # Average loss over the batch

    tf.summary.scalar(name, loss)

    return loss


  # def _compute_loss(self, estimate, target, name):
  #   """Compute the QRDQN loss.
  #   Args:
  #     agent_net: `tf.Tensor`, shape `[None, N]. The tensor output from `self._compute_estimate()`
  #     target_net: `tf.Tensor`, shape `[None, N]. The tensor output from `self._compute_target()`
  #   Returns:
  #     `tf.Tensor` of scalar shape `()`
  #   """
  #   z             = estimate
  #   target_z      = target

  #   # Compute the tensor of mid-quantiles
  #   mid_quantiles = (np.arange(0, self.N, 1, dtype=np.float64) + 0.5) / float(self.N)
  #   mid_quantiles = np.asarray(mid_quantiles, dtype=np.float32)
  #   mid_quantiles = tf.constant(mid_quantiles[None, :], dtype=tf.float32)

  #   def theta_loss(theta, target_thetas):
  #     """
  #     Args:
  #       theta: tf.Tensor, shape `[None, 1]`. Estimated location for a single quantile
  #       target_theta: tf.Tensor, shape `[None, N]`. Target quantile locations
  #     Returns:
  #       tf.Tensor of shape `[None]`
  #     """
  #     td_error      = theta - target_thetas               # out: [None, N]
  #     indicator     = tf.to_float(td_error < 0.0)         # out: [None, N]

  #     # Compute the quantile penalty weights
  #     quant_weight  = mid_quantiles - indicator           # out: [None, N]
  #     # Make sure no gradient flows through the indicator function. The penalty is only a scaling factor
  #     quant_weight  = tf.stop_gradient(quant_weight)

  #     # Pure Quantile Regression Loss
  #     if self.k == 0:
  #       quantile_loss = quant_weight * td_error           # out: [None, N]
  #     # Quantile Huber Loss
  #     else:
  #       quant_weight  = tf.abs(quant_weight)
  #       huber_loss    = tf_utils.huber_loss(td_error, delta=np.float32(self.k))
  #       quantile_loss = quant_weight * huber_loss         # out: [None, N]

  #     # Compute the expected loss for this theta
  #     quantile_loss = tf.reduce_mean(quantile_loss, axis=-1)

  #     return quantile_loss

  #   thetas  = tf.split(z, self.N, axis=-1)                        # Split estimated quantiles
  #   losses  = [theta_loss(theta, target_z) for theta in thetas]   # Compute loss for each quantile
  #   loss    = tf.add_n(losses)                                    # Sum loss over all quantiles
  #   loss    = tf.reduce_mean(loss)                                # Average loss over the batch

  #   tf.summary.scalar(name, loss)

  #   return loss


  def _act_train(self, agent_net, name):
    # Compute the Q-function as expectation of Z; output shape [None, n_actions]
    q       = tf.reduce_mean(agent_net, axis=-1)
    action  = tf.argmax(q, axis=-1, output_type=tf.int32, name=name)

    # Add debugging plot for the variance of the return
    center  = agent_net - tf.expand_dims(q, axis=-1)      # out: [None, n_actions, N]
    z_var   = tf.reduce_mean(tf.square(center), axis=-1)  # out: [None, n_actions]
    tf.summary.scalar("debug/z_var", tf.reduce_mean(z_var))
    tf.summary.histogram("debug/a_rho2", z_var)

    p_a     = tf.identity(action[0],    name="plot/train/a")
    p_q     = tf.identity(q[0],         name="plot/train/q")
    p_z_var = tf.identity(z_var[0],     name="plot/train/z_var")

    self.plot_train["train_actions"] = {
      "a_q":      dict(height=p_q,      a=p_a),
      "a_z_var":  dict(height=p_z_var,  a=p_a),
      # "a_z":      dict(height=p_z,      a=p_a),
    }

    return action


  def _act_eval(self, agent_net, name):
    self.plot_eval["eval_actions"] = dict(self.plot_train["train_actions"])
    return tf.identity(self.a_train, name=name)
