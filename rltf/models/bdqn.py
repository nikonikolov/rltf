import tensorflow as tf

from rltf.models.ddqn import DDQN
from rltf.models.blr  import BLR
from rltf.models      import tf_utils


class BDQN(DDQN):
  """Bayesian Double DQN"""

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, huber_loss, sigma_e, tau, mode="mean"):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
      sigma_e: float. Standard deviation of the noise observation for BLR
      tau: float. Standard deviation for the weight prior in BLR
      huber_loss: bool. Whether to use huber loss or not
    """

    super().__init__(obs_shape, n_actions, opt_conf, gamma, huber_loss)

    self.agent_blr  = [BLR(tau=tau, sigma_e=sigma_e, mode=mode)   for _ in range(self.n_actions)]
    self.target_blr = [BLR(tau=tau, sigma_e=sigma_e, mode="mean") for _ in range(self.n_actions)]

    # Custom TF Tensors and Ops
    self._target    = None    # BLR target
    self._phi       = None    # BLR features
    self.train_blr  = None    # Op for updating the BLR weight posterior
    self.reset_blr  = None    # Op for reseting the BLR to initial weights
    self.a_var      = None    # Tensor with BLR stds


  # def build(self):
  #   super().build()
  #   self.reset_blr = tf.group(*[blr.reset_op for blr in self.agent_blr], name="reset_blr")


  def _conv_nn(self, x):
    """ Build the DQN architecture - as described in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
    Returns:
      `tf.Tensor` of shape `[batch_size, n_actions]`. Contains the Q-function for each action
    """
    with tf.variable_scope("conv_net"):
      # original architecture
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    with tf.variable_scope("action_value"):
      x = tf.layers.dense(x, units=512, activation=tf.nn.relu)
      # Normalize features
      # if self.phi_norm:
      #   x = tf.layers.batch_normalization(phi, axis=-1, training=tf.not_equal(tf.shape(x)[0], 1))
      blrs = self.agent_blr if "agent_net" in tf.get_variable_scope().name else self.target_blr

      # Compute the mean and std prediction from BLR
      blr_out = [blr.apply(x) for blr in blrs]
      # Remember phi and the stds
      if "agent_net" in tf.get_variable_scope().name and self._phi is None:
        self._phi  = x
        self.a_var = tf.concat([std for (_, std) in blr_out], axis=-1)
      # Group the mean predictions
      x = [mean for (mean, _) in blr_out]
      x = tf.concat(x, axis=-1)

    return x


  def _build_train_blr_op(self, phi, target, name):
    """Build the Bayesian Linear Regression ops and estimates
    Args:
      phi: tf.Tensor, shape: `[None, dim_phi]`. The feature tensor
      target: tf.Tensor, as returned by `self._compute_target()`; `[None]`
    Returns:
      tf.Op: The train Op for BLR
    """
    target = tf.expand_dims(target, axis=-1)

    def train_blr(blr, a):
      """Given a BLR instance, select only the examples for the corresponding action"""
      mask = tf.expand_dims(tf.equal(self._act_t_ph, a), axis=-1)
      mask = tf.cast(mask, tf.float32)  # out shape: [None]
      X = phi * mask                    # out shape: [None, dim_phi]
      y = target * mask                 # out shape: [None, 1]
      return blr.train(X, y)

    w_updates = [train_blr(blr, i) for i, blr in enumerate(self.agent_blr)]

    return tf.group(*w_updates, name=name)


  def _compute_target(self, target_net):
    target        = super()._compute_target(target_net)
    self._target  = target
    return target


  def _build_train_op(self, optimizer, loss, agent_vars, name):
    self.train_blr = self._build_train_blr_op(self._phi, self._target, name="train_blr")

    return super()._build_train_op(optimizer, loss, agent_vars, name)



class BDQN_TS(BDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, huber_loss, sigma_e, tau):

    super().__init__(obs_shape, n_actions, opt_conf, gamma, huber_loss, sigma_e, tau, mode="ts")

    # Custom TF Tensors and Ops
    self.reset_ts   = None    # Op that resamples the parameters for TS


  def build(self):
    super().build()

    agent_w   = [blr.w            for blr in self.agent_blr]
    target_w  = [blr.resample_w() for blr in self.target_blr]

    self.reset_ts = tf_utils.assign_vars(agent_w, target_w, name="reset_ts")


  def reset(self, sess):
    sess.run(self.reset_ts)



class BDQN_UCB(BDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, huber_loss, sigma_e, tau, n_stds=0.1):
    """
    Args:
      n_stds: float. Scale constant for the uncertainty
    """
    super().__init__(obs_shape, n_actions, opt_conf, gamma, huber_loss, sigma_e, tau, mode="mean")
    self.n_stds = n_stds       # Number of standard deviations for computing uncertainty


  def _act_train(self, agent_net, name):
    mean    = agent_net
    std     = tf.sqrt(self.a_var)
    action  = tf.argmax(mean + self.n_stds * std, axis=-1, output_type=tf.int32, name=name)

    # Add debug histograms
    tf.summary.histogram("debug/a_std",   std)
    tf.summary.histogram("debug/a_mean",  mean)

    return action



class BDQN_IDS(BDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, huber_loss, sigma_e, tau, policy, n_stds=0.1):
    """
    Args:
      n_stds: float. Scale constant for the uncertainty
    """
    super().__init__(obs_shape, n_actions, opt_conf, gamma, huber_loss, sigma_e, tau, mode="mean")

    assert policy in ["stochastic", "deterministic"]
    self.n_stds = n_stds       # Number of standard deviations for computing uncertainty
    self.rho    = 1.0
    self.policy = policy


  def _act_train(self, agent_net, name):
    mean      = agent_net
    var       = self.a_var
    std       = tf.sqrt(var)
    regret    = tf.reduce_max(mean + self.n_stds * std, axis=-1, keepdims=True)
    regret    = regret - (mean - self.n_stds * std)
    regret_sq = tf.square(regret)
    info_gain = tf.log(1 + var / self.rho**2) + 1e-5
    ids_score = tf.div(regret_sq, info_gain)
    ids_score = tf.check_numerics(ids_score, "IDS score is NaN or Inf")

    if self.policy == "deterministic":
      action    = tf.argmin(ids_score, axis=-1, output_type=tf.int32, name=name)
      a_ucb     = tf.argmax(mean + self.n_stds * std, axis=-1, output_type=tf.int32)
    else:
      scores  = -ids_score    # NOTE: Take -ids_score to make the min have highest probability
      sample  = tf.random_uniform([tf.shape(ids_score)[0], 1], 0.0, 1.0)
      pdf     = scores - tf.expand_dims(tf.reduce_max(scores, axis=-1), axis=-1)
      pdf     = tf.nn.softmax(pdf, axis=-1)
      cdf     = tf.cumsum(pdf, axis=-1, exclusive=True)
      offset  = tf.where(cdf <= sample, tf.zeros_like(cdf), -2*tf.ones_like(cdf))
      sample  = cdf + offset
      action  = tf.argmax(sample, axis=-1, output_type=tf.int32, name=name)
      a_ucb   = None


      a_det   = tf.argmin(ids_score, axis=-1, output_type=tf.int32)
      a_diff  = tf.reduce_mean(tf.cast(tf.equal(a_det, action), tf.float32))
      tf.summary.scalar("debug/a_det_vs_stoch", a_diff)

    # Add debug histograms
    tf.summary.histogram("debug/a_mean",    mean)
    tf.summary.histogram("debug/a_std",     std)
    tf.summary.histogram("debug/a_regret",  regret)
    tf.summary.histogram("debug/a_info",    info_gain)
    tf.summary.histogram("debug/a_ids",     ids_score)

    if a_ucb is not None:
      a_diff_ucb = tf.reduce_mean(tf.cast(tf.equal(a_ucb, action), tf.float32))
      tf.summary.scalar("debug/a_ucb_vs_ids", a_diff_ucb)

    # Set the plottable tensors for video. Use only the first action in the batch
    p_a     = tf.identity(action[0],    name="plot/train/a")
    p_mean  = tf.identity(mean[0],      name="plot/train/mean")
    p_std   = tf.identity(std[0],       name="plot/train/std")
    p_ids   = tf.identity(ids_score[0], name="plot/train/ids")

    self.plot_train["train_actions"] = {
      "a_mean": dict(height=p_mean, a=p_a),
      "a_std":  dict(height=p_std,  a=p_a),
      "a_ids":  dict(height=p_ids,  a=p_a),
    }

    return action


  def _act_eval(self, agent_net, name):
    action = super()._act_eval(agent_net, name)
    # Set the plottable tensors for video. Use only the first action in the batch
    p_a     = tf.identity(action[0],    name="plot/eval/a")
    p_mean  = tf.identity(agent_net[0], name="plot/eval/mean")

    self.plot_eval["eval_actions"] = {
      "a_mean": dict(height=p_mean, a=p_a),
    }

    return action
