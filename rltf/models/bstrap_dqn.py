import tensorflow as tf

from rltf.models  import BaseDQN
from rltf.models  import tf_utils


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
    self._active_head   = None
    self._set_act_head  = None


  def build(self):

    super()._build()

    self._active_head   = tf.Variable([0], trainable=False, name="active_head")
    sample_head         = tf.random_uniform(shape=[1], maxval=self.n_heads, dtype=tf.int32)
    self._set_act_head  = tf.assign(self._active_head, sample_head, name="set_act_head")

    # Preprocess the observation
    self._obs_t   = self._preprocess_obs(self._obs_t_ph)
    self._obs_tp1 = self._preprocess_obs(self._obs_tp1_ph)

    # Construct the Q-network and the target network
    agent_net, x  = self._nn_model(self._obs_t,   scope="agent_net")
    target_net, _ = self._nn_model(self._obs_tp1, scope="target_net")

    # Compute the estimated Q-function and its backup value
    estimate      = self._compute_estimate(agent_net)
    target        = self._compute_target(target_net)

    # Compute the loss
    losses        = self._compute_loss(estimate, target)

    agent_vars    = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_net")
    target_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")

    # Build the optimizer
    optimizer     = self.opt_conf.build()
    # Compute gradients
    grads         = self._compute_grads(losses, x, optimizer)
    # Apply the gradients
    train_op      = optimizer.apply_gradients(grads, name="train_op")

    # Create the Op to update the target
    update_target = tf_utils.assign_vars(target_vars, agent_vars, name="update_target")

    # Compute the train and eval actions
    self.a_train  = self._act_train(agent_net, name="a_train")
    self.a_eval   = self._act_eval(agent_net,  name="a_eval")

    self._train_op      = train_op
    self._update_target = update_target



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
    conv_out = x
    with tf.variable_scope("action_value"):
      heads = [_build_head(x, n_actions) for _ in range(self.n_heads)]
      x = tf.concat(heads, axis=-2)
    return x, conv_out


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
    act_t     = tf.cast(self._act_t_ph, tf.int32)
    act_mask  = tf.one_hot(act_t, self.n_actions, on_value=True, off_value=False, dtype=tf.bool)
    act_mask  = tf.expand_dims(act_mask, axis=-2)
    act_mask  = tf.tile(act_mask, [1, self.n_heads, 1])
    q         = tf.boolean_mask(agent_net, act_mask)
    q         = tf.reshape(q, shape=[-1, self.n_heads])
    return q


  def _compute_target(self, target_net):
    """Compute the Double DQN backup value - use the greedy action from the q estimate
    Args:
      target_net: `tf.Tensor`. The tensor output from `self._nn_model()` for the target
    Returns:
      `tf.Tensor` of shape `[None, n_heads]`
    """
    # Compute the Q-estimate with the agent network variables
    agent_net,_ = self._nn_model(self._obs_tp1, scope="agent_net")

    # Compute the target action
    target_mask = tf.argmax(agent_net, axis=-1, output_type=tf.int32)
    target_mask = tf.one_hot(target_mask, self.n_actions, on_value=True, off_value=False, dtype=tf.bool)

    # Compute the target
    done_mask   = tf.cast(tf.logical_not(self._done_ph), tf.float32)
    done_mask   = tf.expand_dims(done_mask, axis=-1)
    rew_t       = tf.expand_dims(self.rew_t_ph, axis=-1)
    target_q    = tf.boolean_mask(target_net, target_mask)
    target_q    = tf.reshape(target_q, shape=[-1, self.n_heads])
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


  def _compute_grads(self, losses, x_heads, optimizer):
    # Get the conv net and the heads variables
    head_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_net/action_value')
    conv_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_net/conv_net')

    # Compute the gradients of the variables in all heads as well as
    # the sum of gradients backpropagated from each head into the conv net
    head_grads  = tf.gradients(losses, head_vars + [x_heads])

    # Normalize the gradient which is backpropagated the heads to the conv net
    x_heads_g   = head_grads.pop(-1)
    x_heads_g   = x_heads_g / float(self.n_heads)

    # Compute the conv net gradients using chain rule
    conv_grads  = optimizer.compute_gradients(x_heads, conv_vars, grad_loss=x_heads_g)

    # Group grads and apply them
    head_grads  = list(zip(head_grads, head_vars))
    grads       = head_grads + conv_grads

    return grads


  def _restore(self, graph):
    super()._restore()

    self._active_head   = graph.get_tensor_by_name("active_head:0")
    self._set_act_head  = graph.get_operation_by_name("set_act_head")


  def reset(self, sess):
    sess.run(self._set_act_head)
