import tensorflow as tf

from rltf.models  import DQN
from rltf.models  import BayesianLinearRegression
from rltf.models  import tf_utils


class DQN_IDS_BLR(DQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, sigma, tau, phi_norm, same_w,
               huber_loss=True):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
      sigma: float. Standard deviation of the noise observation for BLR and for InfoGain
      tau: float. Standard deviation for the weight prior in BLR
      phi_norm: bool. Whether to normalize the features
      same_w: bool. If True, use the same weights for estimating the Q-function for each action. Else,
        use separate weights as in DQN
      huber_loss: bool. Whether to use huber loss or not
    """

    super().__init__(obs_shape, n_actions, opt_conf, gamma, huber_loss)

    self.dim_phi  = 512
    self.rho      = sigma
    self.n_stds   = 1.0       # Number of standard deviations for computing the regret bound
    self.phi_norm = phi_norm

    blr_params    = dict(sigma=sigma, tau=tau, w_dim=self.dim_phi+1, auto_bias=False)
    if same_w:
      # --------------------------- ARCH: SAME ACTION WEIGHTS ---------------------------
      self.blr        = BayesianLinearRegression(**blr_params)
      self._conv_nn   = self._conv_nn_same_w
      self._dense_nn  = self._dense_nn_same_w
      self._build_blr = self._build_blr_same_w

    else:
      # --------------------------- ARCH: DIFFERENT ACTION WEIGHTS ---------------------------
      self.blr        = [BayesianLinearRegression(**blr_params) for _ in range(self.n_actions)]
      self._conv_nn   = self._conv_nn_diff_w
      self._dense_nn  = self._dense_nn_diff_w
      self._build_blr = self._build_blr_diff_w

    # Custom TF Tensors and Ops
    self._blr_predict = None
    self._target      = None
    self._phi         = None


  # --------------------------- ARCH: DIFFERENT ACTION WEIGHTS ---------------------------

  def _conv_nn_diff_w(self, x):
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
      x = tf.layers.dense(x, units=self.dim_phi, activation=tf.nn.relu)
      if "agent_net" in tf.get_variable_scope().name and self._phi is None: self._phi = x
      x = tf.layers.dense(x, units=n_actions, activation=None)
    return x


  def _dense_nn_diff_w(self, x):
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
      if "agent_net" in tf.get_variable_scope().name and self._phi is None: self._phi = x
      x = tf.layers.dense(x, units=n_actions, activation=None)
    return x


  def _build_blr_diff_w(self, phi, target):
    """Build the Bayesian Linear Regression ops and estimates
    Args:
      phi: tf.Tensor, as returned by `self._nn_model()`
      target: tf.Tensor, as returned by `self._compute_target()`
    Returns:
      tf.Op: The train Op for BLR
      y_means: tf.Tensor, shape `[None, n_actions]`. The mean BLR estimate from `blr.predict()`
      y_sts: tf.Tensor, shape `[None, n_actions]`. The std BLR estimate from `blr.predict()`
    """

    # Normalize features
    if self.phi_norm:
      training  = tf.not_equal(tf.shape(self._obs_t_ph)[0], 1)
      phi       = tf.layers.batch_normalization(phi, axis=-1, training=training)

    # Add bias to the feature transformations
    bias    = tf.ones(shape=[tf.shape(phi)[0], 1], dtype=tf.float32)
    phi     = tf.concat([phi, bias], axis=-1)

    target  = tf.expand_dims(target, axis=-1)

    # Build the Bayesian Regerssion ops and estimates for all actions
    y_means, y_stds, w_updates = [], [], []
    for i in range(self.n_actions):
      self.blr[i].build()
      mask = tf.equal(self._act_t_ph, i)
      X = tf.boolean_mask(phi,    mask)
      y = tf.boolean_mask(target, mask)
      b = tf.shape(X)[0]
      X = tf.cond(tf.not_equal(b, 0), lambda: X, lambda: tf.zeros([1, self.dim_phi+1]))
      y = tf.cond(tf.not_equal(b, 0), lambda: y, lambda: tf.zeros([1, 1]))
      w_update_op = self.blr[i].weight_posterior(X, y)
      y_m, y_std  = self.blr[i].predict(phi)

      w_updates.append(w_update_op)
      y_stds.append(y_std)
      y_means.append(y_m)

    blr_train = tf.group(*w_updates)
    y_means   = tf.concat(y_means, axis=-1)     # output shape [None, n_actions]
    y_stds    = tf.concat(y_stds,  axis=-1)     # output shape [None, n_actions]

    return blr_train, y_means, y_stds


  # --------------------------- ARCH: SAME ACTION WEIGHTS ---------------------------


  def _conv_nn_same_w(self, x):
    """ Build the DQN architecture - as described in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created
    Returns:
      x: `tf.Tensor`, shape `[batch_size, n_actions]`. Contains the Q-function for each action
      phi: `tf.Tensor`, shape `[batch_size, n_actions, dim_phi]`. Contains the state-action features
    """
    with tf.variable_scope("conv_net"):
      # original architecture
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    # Compute the features and the q function for each action
    batch_size = tf.shape(x)[0]
    x_phi = [self._state_action_value(x, a, batch_size) for a in range(self.n_actions)]
    # Concatenate the output Tensors
    x   = tf.concat([x    for x, _    in x_phi], axis=-1)
    phi = tf.concat([phi  for _, phi  in x_phi], axis=1)
    if "agent_net" in tf.get_variable_scope().name and self._phi is None: self._phi = phi
    return x


  def _dense_nn_same_w(self, x):
    """ Build a Neural Network of dense layers only. Used for low-level observations
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created
    Returns:
      x: `tf.Tensor`, shape `[batch_size, n_actions]`. Contains the Q-function for each action
      phi: `tf.Tensor`, shape `[batch_size, n_actions, dim_phi]`. Contains the state-action features
    """
    with tf.variable_scope("dense_net"):
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
    # Compute the features and the q function for each action
    batch_size = tf.shape(x)[0]
    x_phi = [self._state_action_value(x, a, batch_size) for a in range(self.n_actions)]
    # Concatenate the output Tensors
    x   = tf.concat([x    for x, _    in x_phi], axis=-1)
    phi = tf.concat([phi  for _, phi  in x_phi], axis=1)
    if "agent_net" in tf.get_variable_scope().name and self._phi is None: self._phi = phi
    return x


  def _state_action_value(self, x, a, batch_size):
    """
    Returns:
      x: `tf.Tensor`, shape `[batch_size, 1]`. Contains the Q-function for action a
      phi: `tf.Tensor`, shape `[batch_size, 1, dim_phi]`. Contains the features for x and action a
    """
    with tf.variable_scope("action_value", reuse=tf.AUTO_REUSE):
      a = a * tf.ones([batch_size, 1])
      x = tf.concat([x, a], axis=-1)
      x = tf.layers.dense(x, units=self.dim_phi, activation=tf.nn.relu, name="dense_phi")
      phi = tf.expand_dims(x, axis=1)
      x = tf.layers.dense(x, 1, activation=None, name="dense_q")
    return x, phi


  def _build_blr_same_w(self, phi, target):
    """Build the Bayesian Linear Regression ops and estimates
    Args:
      phi: tf.Tensor, as returned by `self._nn_model()`
      target: tf.Tensor, as returned by `self._compute_target()`
    Returns:
      tf.Op: The train Op for BLR
      y_means: tf.Tensor, shape `[None, n_actions]`. The mean BLR estimate from `blr.predict()`
      y_sts: tf.Tensor, shape `[None, n_actions]`. The std BLR estimate from `blr.predict()`
    """

    # Normalize features
    if self.phi_norm:
      raise NotImplementedError()
      # CAREFUL WITH BATCH NORM - HOW SHOULD YOU DO THE NORMALIZATION?
      training  = tf.not_equal(tf.shape(self._obs_t_ph)[0], 1)
      phi       = tf.layers.batch_normalization(phi, axis=-1, training=training)

    # y: Tensor with BLR labels for the batch; shape `[None, 1]`
    y       = tf.expand_dims(target, axis=-1)

    # X: Tensor with BLR train inputs for the batch; shape `[None, phi_dim+1]`
    a_mask  = tf.one_hot(self._act_t_ph, self.n_actions, on_value=True, off_value=False, dtype=tf.bool)
    X       = tf.boolean_mask(phi, a_mask)
    X       = tf.concat([X, tf.ones([tf.shape(phi)[0], 1])], axis=-1)

    # phi: Tensor with BLR test inputs; shape `[n_actions, phi_dim+1]`
    phi     = tf.squeeze(phi, axis=0)         # Assumes batch_size = 1
    phi     = tf.concat([phi, tf.ones([tf.shape(phi)[0], 1])], axis=-1)

    # Build the Bayesian Regression ops and estimates
    self.blr.build()
    blr_train   = self.blr.weight_posterior(X, y)
    y_m, y_std  = self.blr.predict(phi)
    y_means     = tf.transpose(y_m)       # output shape `[1, n_actions]`
    y_stds      = tf.transpose(y_std)     # output shape `[1, n_actions]`

    return blr_train, y_means, y_stds


  # --------------------------- END ARCHS ---------------------------


  def _act_train(self, agent_net, name):
    """
    Args:
      agent_net: list or tuple of two Tensors:
        act_means: shape=`[None, n_actions]` - mean estimate for the q function of each action
        act_stds:  shape=`[None, n_actions]` - std  estimate for the q function of each action
    """
    # DQN Verision
    # return self._act_eval(agent_net, name=name)

    act_means, act_stds = self._blr_predict

    act_regret    = tf.reduce_max(act_means + self.n_stds * act_stds, axis=-1)
    act_regret    = act_regret - (act_means - self.n_stds * act_stds)
    act_regret_sq = tf.square(act_regret)
    act_info_gain = tf.log(1 + tf.square(act_stds) / self.rho**2)
    act_ids       = tf.div(act_regret_sq, act_info_gain)
    act_ids       = tf.argmin(act_ids, axis=-1, output_type=tf.int32, name=name)
    return act_ids


  def _act_eval(self, agent_net, name):
    action = tf.argmax(agent_net, axis=-1, output_type=tf.int32, name=name)
    return action


  def _compute_target(self, target_net):
    target        = super()._compute_target(target_net)
    self._target  = target
    return target


  def _build_train_op(self, optimizer, loss, agent_vars, name):
    # DQN Verision
    # train_op      = optimizer.minimize(loss, var_list=agent_vars, name=name)
    # return train_op

    net_train     = optimizer.minimize(loss, var_list=agent_vars)

    blr_out       = self._build_blr(self._phi, self._target)
    blr_train     = blr_out[0]
    blr_predict   = blr_out[1:]

    self._blr_predict = blr_predict

    # Create weight update op and the train op
    train_op      = tf.group(net_train, blr_train, name=name)

    return train_op
