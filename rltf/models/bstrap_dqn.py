import tensorflow as tf

from rltf.models.dqn  import BaseDQN


class BstrapDQN(BaseDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, huber_loss, n_heads):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
      huber_loss: bool. Whether to use huber loss or not
      n_heads: Number of bootstrap heads
    """

    super().__init__(obs_shape, n_actions, opt_conf, gamma)

    self.huber_loss   = huber_loss
    self.n_heads      = n_heads

    # Custom TF Tensors and Ops
    self._active_head   = None
    self._set_act_head  = None
    self._conv_out      = None


  def build(self):
    self._active_head   = tf.Variable([0], trainable=False, name="active_head")
    sample_head         = tf.random_uniform(shape=[1], maxval=self.n_heads, dtype=tf.int32)
    self._set_act_head  = tf.assign(self._active_head, sample_head, name="set_act_head")

    super().build()


  def _conv_nn(self, x):
    """ Build the Bootstrapped DQN architecture - as described in the original paper
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
    with tf.variable_scope("conv_net"):
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    # Careful: Make sure self._conv_out is set only during the right function call
    if "agent_net" in tf.get_variable_scope().name and self._conv_out is None: self._conv_out = x
    with tf.variable_scope("action_value"):
      heads = [_build_head(x, n_actions) for _ in range(self.n_heads)]
      x = tf.concat(heads, axis=-2)
    return x


  def _act_train(self, agent_net, name):
    # Get the Q function from the active head
    head_mask = tf.one_hot(self._active_head, self.n_heads, on_value=True, off_value=False, dtype=tf.bool)
    q_head    = tf.boolean_mask(agent_net, head_mask)
    # Compute the greedy action
    action    = tf.argmax(q_head, axis=-1, output_type=tf.int32, name=name)
    return action


  def _act_eval(self, agent_net, name):

    def count_value(votes, i):
      count = tf.equal(votes, i)
      count = tf.cast(count, tf.int32)
      count = tf.reduce_sum(count, axis=-1, keep_dims=True)
      return count

    # Get the greedy action from each head; output shape `[batch_size, n_heads]`
    votes   = tf.argmax(agent_net, axis=-1, output_type=tf.int32)
    # Get the action votes; output shape `[batch_size, n_actions]`
    counts  = [count_value(votes, i) for i in range(self.n_actions)]
    counts  = tf.concat(counts, axis=-1)
    # Get the max vote action; output shape `[batch_size]`
    action  = tf.argmax(counts, axis=-1, output_type=tf.int32, name=name)

    return action


  def _compute_estimate(self, agent_net):
    """Get the Q value for the selected action
    Returns:
      `tf.Tensor` of shape `[None, n_heads]`
    """
    a_mask  = tf.one_hot(self._act_t_ph, self.n_actions, dtype=tf.float32)
    a_mask  = tf.tile(tf.expand_dims(a_mask, axis=-2), [1, self.n_heads, 1])
    q       = tf.reduce_sum(agent_net * a_mask, axis=-1)
    return q


  def _compute_target(self, target_net):
    """Compute the Double DQN backup value - use the greedy action from the q estimate
    Args:
      target_net: `tf.Tensor`. The tensor output from `self._nn_model()` for the target
    Returns:
      `tf.Tensor` of shape `[None, n_heads]`
    """
    # Compute the Q-estimate with the agent network variables
    agent_net   = self._nn_model(self._obs_tp1, scope="agent_net")

    # Compute the target action
    target_act  = tf.argmax(agent_net, axis=-1, output_type=tf.int32)
    target_mask = tf.one_hot(target_act, self.n_actions, dtype=tf.float32)

    # Compute the target
    done_mask   = tf.cast(tf.logical_not(self._done_ph), tf.float32)
    done_mask   = tf.expand_dims(done_mask, axis=-1)
    rew_t       = tf.expand_dims(self.rew_t_ph, axis=-1)
    target_q    = tf.reduce_sum(target_net * target_mask, axis=-1)
    target_q    = rew_t + self.gamma * done_mask * target_q
    target_q    = tf.stop_gradient(target_q)

    return target_q


  def _compute_loss(self, estimate, target):
    """
    Args: shape `[None, n_heads]`
    Returns:
      List of size `n_heads` with a scalar tensor loss for each head
    """
    if self.huber_loss:
      loss = tf.losses.huber_loss(target, estimate, reduction=tf.losses.Reduction.NONE)
    else:
      loss = tf.losses.mean_squared_error(target, estimate, reduction=tf.losses.Reduction.NONE)

    losses = tf.split(loss, self.n_heads, axis=-1)
    losses = [tf.reduce_mean(loss) for loss in losses]

    tf.summary.scalar("train/loss", tf.add_n(losses)/self.n_heads)

    return losses


  def _build_train_op(self, optimizer, loss, agent_vars, name):
    x_heads = self._conv_out

    # Get the conv net and the heads variables
    head_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_net/action_value')
    conv_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_net/conv_net')

    # Compute the gradients of the variables in all heads as well as
    # the sum of gradients backpropagated from each head into the conv net
    head_grads  = tf.gradients(loss, head_vars + [x_heads])

    # Normalize the gradient which is backpropagated the heads to the conv net
    x_heads_g   = head_grads.pop(-1)
    x_heads_g   = x_heads_g / float(self.n_heads)

    # Compute the conv net gradients using chain rule
    conv_grads  = optimizer.compute_gradients(x_heads, conv_vars, grad_loss=x_heads_g)

    # Group grads and apply them
    head_grads  = list(zip(head_grads, head_vars))
    grads       = head_grads + conv_grads
    train_op    = optimizer.apply_gradients(grads, name=name)

    return train_op


  def _restore(self, graph):
    super()._restore(graph)

    self._active_head   = graph.get_tensor_by_name("active_head:0")
    self._set_act_head  = graph.get_operation_by_name("set_act_head")


  def reset(self, sess):
    sess.run(self._set_act_head)
