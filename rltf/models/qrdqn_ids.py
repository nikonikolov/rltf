import tensorflow as tf

from rltf.models    import DQN_IDS
from rltf.models    import QRDQN
from rltf.tf_utils  import tf_utils


class QRDQN_IDS(DQN_IDS, QRDQN):

  def __init__(self, **kwargs):
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
    super().__init__(**kwargs)

    # Custom TF Tensors and Ops
    self.rho2   = None


  def _conv_nn(self, x):
    """ Build the Bootstrapped DQN architecture - as described in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
    Returns:
      Tuple of `tf.Tensor`s. First tensor is of shape `[batch_size, n_heads, n_actions]` and contains the
      Q-function bootstrapped estimates. Second tensor is of shape `[batch_size, n_actions, N]` and
      contains the QRDQN return distribution for each action
    """
    n_actions = self.n_actions
    N         = self.N
    # k_init    = tf_utils.init_dqn
    k_init    = tf_utils.init_glorot_normal
    # k_init    = tf_utils.init_default

    def build_bstrap_head(x):
      """ Build the head of the DQN network
      Args:
        x: tf.Tensor. Tensor for the input
      Returns:
        `tf.Tensor` of shape `[batch_size, 1, n_actions]`. Contains the Q-function for each action
      """
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
      x = tf.layers.dense(x, units=n_actions, activation=None)
      x = tf.expand_dims(x, axis=-2)
      return x

    def build_z_head(x):
      """ Build the head of the QRDQN network
      Args:
        x: tf.Tensor. Tensor for the input
      Returns:
        `tf.Tensor` of shape `[batch_size, n_actions, N]`. Contains the Q-function distribution
          for each action
      """
      x = tf.layers.dense(x, 512,         activation=tf.nn.relu,  kernel_initializer=k_init())
      x = tf.layers.dense(x, N*n_actions, activation=None,        kernel_initializer=k_init())
      x = tf.reshape(x, [-1, n_actions, N])
      return x

    with tf.variable_scope("conv_net"):
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    x = tf.layers.flatten(x)

    # Careful: Make sure self._conv_out is set only during the right function call
    if "agent_net" in tf.get_variable_scope().name and self._conv_out is None: self._conv_out = x

    # Build the QRDQN head
    with tf.variable_scope("distribution_value"):
      z = build_z_head(x)

    # Build the Bootstrap heads
    with tf.variable_scope("action_value"):
      heads = [build_bstrap_head(x) for _ in range(self.n_heads)]
      x = tf.concat(heads, axis=-2)
    return dict(q_values=x, quantiles=z)


  def _compute_estimate(self, agent_net):
    """Get the Q value for the selected action
    Args:
      agent_net: tuple of `tf.Tensor`s. Output from the agent network. Shapes:
        `[batch_size, n_heads, n_actions]` and `[batch_size, n_actions, N]`
    Returns:
      Tuple of `tf.Tensor`s of shapes `[batch_size, n_heads]` and `[batch_size, N]`
    """
    q, z = agent_net["q_values"], agent_net["quantiles"]
    q = DQN_IDS._compute_estimate(self, q)        # out: [None, n_heads]
    z = QRDQN._compute_estimate(self, z)          # out: [None, N]
    return dict(q_values=q, quantiles=z)


  def _select_target(self, target_net):
    """Select the Double DQN target
    Args:
      target_net: `tf.Tensor`. shape `[None, n_heads, n_actions]. The output from `self._nn_model()`
        for the target
    Returns:
      `tf.Tensor` of shape `[None, n_heads]`
    """
    n_actions   = self.n_actions

    # Compute the Q-estimate with the agent network variables and select the maximizing action
    agent_net   = self._nn_model(self.obs_tp1, scope="agent_net")       # out: [None, n_heads, n_actions]
    agent_net   = agent_net["q_values"]  # Select only the Q-tensor
    target_act  = tf.argmax(agent_net, axis=-1, output_type=tf.int32)   # out: [None, n_heads]

    # Select the target Q-function
    target_mask = tf.one_hot(target_act, n_actions, dtype=tf.float32)   # out: [None, n_heads, n_actions]
    target_q    = tf.reduce_sum(target_net * target_mask, axis=-1)      # out: [None, n_heads]

    return target_q


  def _compute_target(self, target_net):
    """Compute the backups
    Args:
      target_net: tuple of `tf.Tensor`s. Output from the target network. Shapes:
        `[batch_size, n_heads, n_actions]` and `[batch_size, n_actions, N]`
    Returns:
      Tuple of `tf.Tensor`s of shapes `[batch_size, n_heads]` and `[batch_size, N]`
    """
    target_q, target_z = target_net["q_values"], target_net["quantiles"]
    # DQN_IDS call to self._select_target resolves to the QRDQN_IDS._select_target()
    backup_q = DQN_IDS._compute_target(self, target_q)
    # NOTE: Do NOT call QRDQN._compute_target(self, target_z) - call to self._select_target()
    # will resolve to QRDQN_IDS._select_target() - incorrect
    target_z = QRDQN._select_target(self, target_z)
    backup_z = QRDQN._compute_backup(self, target_z)
    backup_z = tf.stop_gradient(backup_z)
    return dict(target_q=backup_q, target_z=backup_z)


  def _compute_loss(self, estimate, target, name):
    q, z                = estimate["q_values"], estimate["quantiles"]
    target_q, target_z  = target["target_q"], target["target_z"]


    head_loss = DQN_IDS._compute_loss(self, q, target_q, name)
    z_loss    = QRDQN._compute_loss(self, z, target_z, "train/z_loss")

    return dict(head_loss=head_loss, z_loss=z_loss)


  def _build_train_op(self, optimizer, loss, agent_vars, name):
    head_loss = loss["head_loss"]
    z_loss    = loss["z_loss"]

    # Get the bootsrapped heads and conv net train op
    train_net = DQN_IDS._build_train_op(self, optimizer, head_loss, agent_vars, name=None)

    # Get the train op for the distributional FC layers
    z_vars    = tf_utils.scope_vars(agent_vars, scope='agent_net/distribution_value')
    train_z   = QRDQN._build_train_op(self, optimizer, z_loss, z_vars, name=None)

    train_op  = tf.group(train_net, train_z, name=name)

    return train_op


  # Propagate QR loss gradients
  # def _build_train_op(self, optimizer, loss, agent_vars, name):
  #   head_loss = loss["head_loss"]
  #   z_loss    = loss["z_loss"]

  #   # Update the Bootstrap heads variables. Do not backpropagate gradients to the conv layers
  #   head_vars   = tf_utils.scope_vars(agent_vars, scope='agent_net/action_value')
  #   head_grads  = tf.gradients(loss, head_vars)
  #   head_grads  = list(zip(head_grads, head_vars))
  #   train_heads = optimizer.apply_gradients(head_grads)
  #   # heads_grads = optimizer.compute_gradients(head_loss, var_list=head_vars)
  #   # train_heads = optimizer.minimize(head_loss, var_list=head_vars)

  #   # Update the conv and the QRDQN head variables based on QRDQN loss
  #   conv_vars = tf_utils.scope_vars(agent_vars, scope='agent_net/conv_net')
  #   z_vars    = tf_utils.scope_vars(agent_vars, scope='agent_net/distribution_value')
  #   train_z   = optimizer.minimize(z_loss, var_list=conv_vars+z_vars)

  #   train_op  = tf.group(train_heads, train_z, name=name)
  #   return train_op


  def _act_train(self, agent_net, name):
    # agent_net tuple of shapes: [None, n_heads, n_actions], [None, n_actions, N]

    z_var     = self._compute_z_variance(agent_net["quantiles"], normalize=True)  # [None, n_actions]
    self.rho2 = tf.maximum(z_var, 0.25)

    action    = DQN_IDS._act_train(self, agent_net["q_values"], name)

    # Add debugging data for TB
    tf.summary.histogram("debug/a_rho2", self.rho2)
    tf.summary.scalar("debug/z_var", tf.reduce_mean(z_var))

    # Append the plottable tensors for episode recordings
    p_rho2  = tf.identity(self.rho2[0], name="plot/train/rho2")
    p_a     = self.plot_conf.true_train_spec["train_actions"]["a_mean"]["a"]
    self.plot_conf.true_train_spec["train_actions"]["a_rho2"] = dict(height=p_rho2, a=p_a)

    return action


  def _act_eval(self, agent_net, name):
    return DQN_IDS._act_eval(self, agent_net["q_values"], name)
