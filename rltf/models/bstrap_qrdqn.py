import numpy as np
import tensorflow as tf

from rltf.models.bstrap_dqn import BaseBstrapDQN
from rltf.models.bstrap_dqn import BstrapDQN_IDS
from rltf.models            import tf_utils


class BaseBstrapQRDQN(BaseBstrapDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, n_heads, N, k):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
      n_heads: Number of bootstrap heads
      N: int. number of quantiles
      k: int. Huber loss order
    """
    super().__init__(obs_shape, n_actions, opt_conf, gamma, True, n_heads)

    self.n_heads  = n_heads
    self.N        = N
    self.k        = k


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
    init_glorot_normal = tf_utils.init_glorot_normal

    def build_head(x):
      """ Build the head of the QRDQN network
      Args:
        x: tf.Tensor. Tensor for the input
      Returns:
        `tf.Tensor` of shape `[batch_size, 1, n_actions, N]`. Contains the Q-function distribution
          for each action
      """
      x = tf.layers.dense(x, 512,         activation=tf.nn.relu,  kernel_initializer=init_glorot_normal())
      x = tf.layers.dense(x, N*n_actions, activation=None,        kernel_initializer=init_glorot_normal())
      x = tf.reshape(x, [-1, n_actions, N])
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
    # a_mask  = tf.expand_dims(a_mask, axis=1)              # out: [None, 1,       n_actions]
    # a_mask  = tf.tile(a_mask, [1, self.n_heads, 1])       # out: [None, n_heads, n_actions]
    # a_mask  = tf.expand_dims(a_mask, axis=-1)             # out: [None, n_heads, n_actions, 1]
    z       = tf.reduce_sum(agent_net * a_mask, axis=-2)        # out: [None, n_heads, N]
    return z


  def _compute_target(self, target_net):
    """Compute the backup value
    Args:
      target_net: `tf.Tensor`. The tensor output from `self._nn_model()` for the target
    Returns:
      `tf.Tensor` of shape `[None, n_heads, N]`
    """

    # # Compute the Double Q-estimate tartget
    # agent_net   = self._nn_model(self._obs_tp1, scope="agent_net")    # out: [None, n_heads, n_actions, N]
    # target_q    = tf.reduce_mean(agent_net, axis=-1)                  # out: [None, n_heads, n_actions]

    # Compute the target q function and the greedy action
    target_z    = target_net
    target_q    = tf.reduce_mean(target_z, axis=-1)                   # out: [None, n_heads, n_actions]
    target_act  = tf.argmax(target_q, axis=-1, output_type=tf.int32)  # out: [None, n_heads]
    target_mask = tf.one_hot(target_act, self.n_actions, dtype=tf.float32)  # out: [None, n_heads, n_actions]
    target_mask = tf.expand_dims(target_mask, axis=-1)                # out: [None, n_heads, n_actions, 1]
    target_z    = tf.reduce_sum(target_net * target_mask, axis=-2)    # out: [None, n_heads, N]

    # Compute the target
    done_mask   = tf.cast(tf.logical_not(self._done_ph), tf.float32)  # out: [None]
    done_mask   = tf.reshape(done_mask, [-1, 1, 1])                   # out: [None, 1, 1]
    done_mask   = tf.tile(done_mask, [1, self.n_heads, 1])            # out: [None, n_heads, 1]
    rew_t       = tf.reshape(self.rew_t_ph, [-1, 1, 1])               # out: [None, 1, 1]
    rew_t       = tf.tile(rew_t, [1, self.n_heads, 1])                # out: [None, n_heads, 1]
    target_z    = rew_t + self.gamma * done_mask * target_z           # out: [None, n_heads, N]
    target_z    = tf.stop_gradient(target_z)

    return target_z


  def _compute_loss(self, estimate, target):
    """
    Args:
      estimate: tf.Tensor, shape `[None, n_heads, N]`. Q-function estimate
      target: tf.Tensor, shape `[None, n_heads, N]`. Q-function target
    Returns:
      List of size `n_heads` with a scalar tensor loss for each head
    """

    # Compute the tensor of mid-quantiles
    mid_quantiles = (np.arange(0, self.N, 1, dtype=np.float32) + 0.5) / float(self.N)
    mid_quantiles = tf.constant(mid_quantiles[None, None, :], dtype=tf.float32)   # out: [1, 1, N]

    estimates     = tf.split(estimate, self.n_heads, axis=1)    # list of shapes [None, 1, N]
    targets       = tf.split(target,   self.n_heads, axis=1)    # list of shapes [None, 1, N]
    estimates     = [tf.squeeze(e, axis=1) for e in estimates]  # list of shapes [None, N]
    targets       = [tf.squeeze(t, axis=1) for t in targets]    # list of shapes [None, N]

    def compute_head_loss(z, target_z):
      """
      Args:
        z: tf.Tensor, shape `[None, N]`
        target_z: tf.Tensor, shape `[None, N]`
      Returns:
        tf.Tensor of scalar shape `()`
      """
      # Operate over last dimensions to get result for for theta_i
      z_diff        = tf.expand_dims(target_z, axis=-2) - tf.expand_dims(z, axis=-1)  # out: [None, N, N]
      indicator_fn  = tf.to_float(z_diff < 0.0)                                       # out: [None, N, N]
      penalty_w     = mid_quantiles - indicator_fn                                    # out: [None, N, N]

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
      return loss

    losses = [compute_head_loss(z, target_z) for z, target_z in zip(estimates, targets)]

    tf.summary.scalar("train/loss", tf.add_n(losses)/self.n_heads)

    return losses



class BstrapQRDQN(BaseBstrapQRDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, n_heads, N, k):
    super().__init__(obs_shape, n_actions, opt_conf, gamma, n_heads, N, k)

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
    # Get the Z distribution from the active head
    head_mask = tf.one_hot(self._active_head, self.n_heads, dtype=tf.float32) # out: [1,    n_heads]
    head_mask = tf.tile(head_mask, [tf.shape(agent_net)[0], 1])               # out: [None, n_heads]
    head_mask = tf.reshape(head_mask, [-1, self.n_heads, 1, 1])               # out: [None, n_heads, 1, 1]
    z         = tf.reduce_sum(agent_net * head_mask, axis=1)                  # out: [None, n_actions, N]

    # Compute the greedy action
    q       = tf.reduce_mean(z, axis=-1)                                      # out: [None, n_actions]
    action  = tf.argmax(q, axis=-1, output_type=tf.int32, name=name)          # out: [None]
    return action


  def _act_eval(self, agent_net, name):
    q = tf.reduce_mean(agent_net, axis=-1)  # out: [1, n_heads, n_actions]
    return self._act_eval_vote(q, name)


  def reset(self, sess):
    sess.run(self._set_act_head)


  def _restore(self, graph):
    super()._restore(graph)
    self._active_head   = graph.get_tensor_by_name("active_head:0")
    self._set_act_head  = graph.get_operation_by_name("set_act_head")



class BstrapQRDQN_IDS(BaseBstrapQRDQN):
  """IDS policy from Boostrapped QRDQN"""

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, n_heads, N, k, policy, n_stds=0.1):
    super().__init__(obs_shape, n_actions, opt_conf, gamma, n_heads, N, k)

    assert policy in ["stochastic", "deterministic"]
    self.n_stds = n_stds    # Number of standard deviations for computing uncertainty
    self.policy = policy
    self.rho2   = None


  def _act_train(self, agent_net, name):
    # agent_net shape: [None, n_heads, n_actions, N]

    q_heads   = tf.reduce_mean(agent_net, axis=-1)            # out: [None, n_heads, n_actions]

    # Estimate return variance
    z_center  = agent_net - tf.expand_dims(q_heads, axis=-1)  # out: [None, n_heads, n_actions, N]
    z_var     = tf.reduce_mean(tf.square(z_center), axis=-1)  # out: [None, n_heads, n_actions]
    rho2      = tf.reduce_mean(z_var, axis=1)                 # out: [None, n_actions]
    self.rho2 = rho2 + 1e-5
    # self.rho2   = 0.5**2    # Const for IDS Info Gain

    # Compute the IDS action - ugly way
    action    = BstrapDQN_IDS._act_train(self, q_heads, name)

    # Add debugging data for  TB
    tf.summary.histogram("debug/a_rho2", self.rho2)
    tf.summary.scalar("debug/rho2", tf.reduce_mean(self.rho2))

    p_rho2  = tf.identity(self.rho2[0], name="plot/train/rho2")
    p_a     = self.plot_train["train_actions"]["a_mean"]["a"]

    self.plot_train["train_actions"]["a_rho2"] = dict(height=p_rho2,  a=p_a)

    return action


  def _act_eval(self, agent_net, name):
    q = tf.reduce_mean(agent_net, axis=-1)    # out: [1, n_heads, n_actions]
    # return self._act_eval_greedy(q, name)
    return self._act_eval_vote(q, name)


  def _restore(self, graph):
    # Ulgy way to restore IDS data
    BstrapDQN_IDS._restore(self, graph)

    a     = graph.get_tensor_by_name("plot/train/a:0")
    rho2  = graph.get_tensor_by_name("plot/train/rho2:0")

    self.plot_train["train_actions"]["a_rho2"] = dict(height=rho2, a=a)
