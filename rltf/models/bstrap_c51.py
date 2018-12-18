import tensorflow as tf

from rltf.models  import BstrapDQN_IDS
from rltf.models  import C51
from rltf.models  import tf_utils


class BstrapC51_IDS(BstrapDQN_IDS, C51):

  def __init__(self, **kwargs):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
      n_heads: Number of bootstrap heads
      huber_loss: bool. Use huber loss for the bootstrap heads
      V_min: float. lower bound for histrogram range
      V_max: float. upper bound for histrogram range
      N: int. number of histogram bins
      n_stds: float. Standard deviation scale for computing regret
    """
    super().__init__(**kwargs)

    # Custom TF Tensors and Ops
    self.rho2   = None

    # Use C51 algo projection and log loss
    C51._project_distribution = C51._project_distribution_algo
    C51._compute_loss = C51._compute_loss_algo


  def _conv_nn(self, x):
    """ Build the Bootstrapped DQN architecture - as described in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
    Returns:
      Tuple of `tf.Tensor`s. First tensor is of shape `[batch_size, n_heads, n_actions]` and contains the
      Q-function bootstrapped estimates. Second tensor is of shape `[batch_size, n_actions, N]` and
      contains the C51 return distribution for each action
    """
    n_actions = self.n_actions
    N         = self.N

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
      """ Build the head of the C51 network
      Args:
        x: tf.Tensor. Tensor for the input
      Returns:
        `tf.Tensor` of shape `[batch_size, n_actions, N]`. Contains the Q-function distribution
          for each action
      """
      x = tf.layers.dense(x, 512,         activation=tf.nn.relu)
      x = tf.layers.dense(x, N*n_actions, activation=None)
      x = tf.reshape(x, [-1, n_actions, N])
      # Compute Softmax probabilities in numerically stable way
      p = tf_utils.softmax(x, axis=-1)
      return x, p

    with tf.variable_scope("conv_net"):
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    x = tf.layers.flatten(x)

    # Careful: Make sure self._conv_out is set only during the right function call
    if "agent_net" in tf.get_variable_scope().name and self._conv_out is None: self._conv_out = x

    # Build the C51 head
    with tf.variable_scope("distribution_value"):
      l, p = build_z_head(x)

    # Build the Bootstrap heads
    with tf.variable_scope("action_value"):
      heads = [build_bstrap_head(x) for _ in range(self.n_heads)]
      x = tf.concat(heads, axis=-2)
    return dict(q_values=x, logits=l, softmax=p)


  def _compute_estimate(self, agent_net):
    """Get the Q value for the selected action
    Args:
      agent_net: tuple of `tf.Tensor`s. Output from the agent network. Shapes:
        `[batch_size, n_heads, n_actions]` and `[batch_size, n_actions, N]`
    Returns:
      Tuple of `tf.Tensor`s of shapes `[batch_size, n_heads]` and `[batch_size, N]`
    """
    q, z = agent_net["q_values"], agent_net["logits"]
    q = BstrapDQN_IDS._compute_estimate(self, q)  # out: [None, n_heads]
    z = C51._compute_estimate(self, z)            # logits; out: [None, N]
    return dict(q_values=q, logits=z)


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
    target_q, target_z = target_net["q_values"], target_net["logits"]
    # BstrapDQN_IDS call to self._select_target resolves to the BstrapC51_IDS._select_target()
    backup_q = BstrapDQN_IDS._compute_target(self, target_q)
    # NOTE: Do NOT call C51._compute_target(self, target_z) - call to self._select_target()
    # will resolve to BstrapC51_IDS._select_target() - incorrect
    target_z = C51._select_target(self, target_z)
    backup_z = C51._compute_backup(self, target_z)
    backup_z = tf.stop_gradient(backup_z)
    return dict(target_q=backup_q, target_p=backup_z)


  def _compute_loss(self, estimate, target, name):
    q, logits_z         = estimate["q_values"], estimate["logits"]
    target_q, target_p  = target["target_q"], target["target_p"]

    head_loss = BstrapDQN_IDS._compute_loss(self, q, target_q, name)
    z_loss    = C51._compute_loss(self, logits_z, target_p, "train/z_loss")

    return dict(head_loss=head_loss, z_loss=z_loss)


  def _build_train_op(self, optimizer, loss, agent_vars, name):
    head_loss = loss["head_loss"]
    z_loss    = loss["z_loss"]

    # Get the Bootsrapped heads and conv net train op
    train_net = BstrapDQN_IDS._build_train_op(self, optimizer, head_loss, agent_vars, name=None)

    # Get the train op for the distributional FC layers
    z_vars    = tf_utils.scope_vars(agent_vars, scope='agent_net/distribution_value')
    train_z   = C51._build_train_op(self, optimizer, z_loss, z_vars, name=None)

    train_op  = tf.group(train_net, train_z, name=name)

    return train_op


  def _act_train(self, agent_net, name):
    # agent_net tuple of shapes: [None, n_heads, n_actions], [None, n_actions, N]

    z_var     = self._compute_z_variance(logits=agent_net["logits"], normalize=True)  # [None, n_actions]
    # z_var     = self._compute_z_variance(z=agent_net["softmax"], normalize=True)  # [None, n_actions]
    self.rho2 = tf.maximum(z_var, 0.25)

    action    = BstrapDQN_IDS._act_train(self, agent_net["q_values"], name)

    # Add debugging data for TB
    tf.summary.histogram("debug/a_rho2", self.rho2)
    tf.summary.scalar("debug/z_var", tf.reduce_mean(z_var))

    # Append the plottable tensors for episode recordings
    p_rho2  = tf.identity(self.rho2[0], name="plot/train/rho2")
    p_a     = self.plot_conf.true_train_spec["train_actions"]["a_mean"]["a"]
    self.plot_conf.true_train_spec["train_actions"]["a_rho2"] = dict(height=p_rho2, a=p_a)

    return action


  def _act_eval(self, agent_net, name):
    return BstrapDQN_IDS._act_eval(self, agent_net["q_values"], name)
