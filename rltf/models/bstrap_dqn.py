import tensorflow as tf

from rltf.models.dqn  import BaseDQN
from rltf.models      import tf_utils


class BaseBstrapDQN(BaseDQN):

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

    self.huber_loss = huber_loss
    self.n_heads    = n_heads

    # Custom TF Tensors and Ops
    self._conv_out  = None


  def _conv_nn(self, x):
    """ Build the Bootstrapped DQN architecture - as described in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
    Returns:
      `tf.Tensor` of shape `[batch_size, n_heads, n_actions]`. Contains the Q-function for each action
    """

    def _build_head(x, n_actions):
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
    head_vars   = tf_utils.scope_vars(agent_vars, scope='agent_net/action_value')
    conv_vars   = tf_utils.scope_vars(agent_vars, scope='agent_net/conv_net')

    # Compute the gradients of the variables in all heads as well as
    # the sum of gradients backpropagated from each head into the conv net
    head_grads  = tf.gradients(loss, head_vars + [x_heads])

    # Normalize the gradient which is backpropagated the heads to the conv net
    x_heads_g   = head_grads.pop(-1)
    x_heads_g   = x_heads_g / float(self.n_heads)

    # Compute the conv net gradients using chain rule
    if conv_vars:
      conv_grads = optimizer.compute_gradients(x_heads, conv_vars, grad_loss=x_heads_g)
    else:
      conv_grads = []

    # Group grads and apply them
    head_grads  = list(zip(head_grads, head_vars))
    grads       = head_grads + conv_grads
    train_op    = optimizer.apply_gradients(grads, name=name)

    return train_op


  def reset(self, sess):
    pass


  def _act_eval_vote(self, agent_net, name):
    """Evaluation action based on voting policy from the heads"""

    def count_value(votes, i):
      count = tf.equal(votes, i)
      count = tf.cast(count, tf.int32)
      count = tf.reduce_sum(count, axis=-1, keepdims=True)
      return count

    # Get the greedy action from each head; output shape `[batch_size, n_heads]`
    votes   = tf.argmax(agent_net, axis=-1, output_type=tf.int32)
    # Get the action votes; output shape `[batch_size, n_actions]`
    votes   = [count_value(votes, i) for i in range(self.n_actions)]
    votes   = tf.concat(votes, axis=-1)
    # Get the max vote action; output shape `[batch_size]`
    action  = tf.argmax(votes, axis=-1, output_type=tf.int32, name=name)

    # Set the plottable tensors for video. Use only the first action in the batch
    p_a     = tf.identity(action[0],  name="plot/eval/a")
    p_vote  = tf.identity(votes[0],   name="plot/eval/vote")

    self.plot_eval["eval_actions"] = {
      "a_vote": dict(height=p_vote, a=p_a),
    }

    return action


  def _act_eval_greedy(self, agent_net, name):
    """Evaluation action based on the greedy action w.r.t. the mean of all heads"""

    mean    = tf.reduce_mean(agent_net, axis=1)
    action  = tf.argmax(mean, axis=-1, output_type=tf.int32, name=name)

    # Set the plottable tensors for video. Use only the first action in the batch
    p_a     = tf.identity(action[0],  name="plot/eval/a")
    p_mean  = tf.identity(mean[0],    name="plot/eval/mean")

    self.plot_eval["eval_actions"] = {
      "a_mean": dict(height=p_mean, a=p_a),
    }

    return action


  def _restore_act_eval_plot(self, graph):
    a = graph.get_tensor_by_name("plot/eval/a:0")

    try:
      vote = graph.get_tensor_by_name("plot/eval/vote:0")
      self.plot_eval["eval_actions"] = {
        "a_vote": dict(height=vote, a=a),
      }
    except KeyError:
      mean = graph.get_tensor_by_name("plot/eval/mean:0")
      self.plot_eval["eval_actions"] = {
        "a_mean": dict(height=mean, a=a),
      }



class BstrapDQN(BaseBstrapDQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, huber_loss, n_heads):

    super().__init__(obs_shape, n_actions, opt_conf, gamma, huber_loss, n_heads)

    # Custom TF Tensors and Ops
    self._active_head   = None
    self._set_act_head  = None


  def build(self):
    self._active_head   = tf.Variable([0], trainable=False, name="active_head")
    sample_head         = tf.random_uniform(shape=[1], maxval=self.n_heads, dtype=tf.int32)
    self._set_act_head  = tf.assign(self._active_head, sample_head, name="set_act_head")
    super().build()


  def _act_train(self, agent_net, name):
    """Select the greedy action from the selected head
    Args:
      agent_net: `tf.Tensor`, shape `[None, n_heads, n_actions]. The tensor output from
        `self._nn_model()` for the agent
    Returns:
      `tf.Tensor` of shape `[None]`
    """
    # Get the Q function from the active head
    head_mask = tf.one_hot(self._active_head, self.n_heads, dtype=tf.float32) # out: [1,    n_heads]
    head_mask = tf.tile(head_mask, [tf.shape(agent_net)[0], 1])               # out: [None, n_heads]
    head_mask = tf.expand_dims(head_mask, axis=-1)                            # out: [None, n_heads, 1]
    q_head    = tf.reduce_sum(agent_net * head_mask, axis=1)                  # out: [None, n_actions]

    # Compute the greedy action
    action    = tf.argmax(q_head, axis=-1, output_type=tf.int32, name=name)
    return action


  def _act_eval(self, agent_net, name):
    return self._act_eval_vote(agent_net, name)


  def reset(self, sess):
    sess.run(self._set_act_head)


  def _restore(self, graph):
    super()._restore(graph)
    self._active_head   = graph.get_tensor_by_name("active_head:0")
    self._set_act_head  = graph.get_operation_by_name("set_act_head")



class BstrapDQN_UCB(BaseBstrapDQN):
  """UCB policy from Boostrapped DQN"""

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, huber_loss, n_heads, n_stds=0.1):
    super().__init__(obs_shape, n_actions, opt_conf, gamma, huber_loss, n_heads)
    self.n_stds = n_stds       # Number of standard deviations for computing uncertainty


  def _act_train(self, agent_net, name):
    mean    = tf.reduce_mean(agent_net, axis=1)
    std     = agent_net - tf.expand_dims(mean, axis=-2)
    std     = tf.sqrt(tf.reduce_mean(tf.square(std), axis=1))
    action  = tf.argmax(mean + self.n_stds * std, axis=-1, output_type=tf.int32, name=name)

    # Add debug histograms
    tf.summary.histogram("debug/a_std",   std)
    tf.summary.histogram("debug/a_mean",  mean)

    return action


  def _act_eval(self, agent_net, name):
    return self._act_eval_vote(agent_net, name)



class BstrapDQN_Ensemble(BaseBstrapDQN):
  """Ensemble policy from Boostrapped DQN"""

  def _act_train(self, agent_net, name):
    # TODO: If plotting, self.plot_train will be empty
    return self._act_eval_vote(agent_net, name)


  def _act_eval(self, agent_net, name):
    return tf.identity(self.a_train, name=name)



class BstrapDQN_IDS(BaseBstrapDQN):
  """IDS policy from Boostrapped DQN"""

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, huber_loss, n_heads, policy, n_stds=0.1):
    super().__init__(obs_shape, n_actions, opt_conf, gamma, huber_loss, n_heads)

    assert policy in ["stochastic", "deterministic"]
    self.n_stds = n_stds    # Number of standard deviations for computing uncertainty
    self.rho2   = 1.0**2    # Const for IDS Info Gain
    self.policy = policy


  def _act_train(self, agent_net, name):
    mean      = tf.reduce_mean(agent_net, axis=1)
    zero_mean = agent_net - tf.expand_dims(mean, axis=-2)
    var       = tf.reduce_mean(tf.square(zero_mean), axis=1)
    std       = tf.sqrt(var)
    regret    = tf.reduce_max(mean + self.n_stds * std, axis=-1, keepdims=True)
    regret    = regret - (mean - self.n_stds * std)
    regret_sq = tf.square(regret)
    info_gain = tf.log(1 + var / self.rho2) + 1e-5
    ids_score = tf.div(regret_sq, info_gain)
    ids_score = tf.check_numerics(ids_score, "IDS score is NaN or Inf")

    if self.policy == "deterministic":
      action  = tf.argmin(ids_score, axis=-1, output_type=tf.int32, name=name)
      a_ucb   = tf.argmax(mean + self.n_stds * std, axis=-1, output_type=tf.int32)
    else:
      # Sample via categorical distribution
      scores  = -ids_score    # NOTE: Take -ids_score to make the min have highest probability
      sample  = tf.random_uniform([tf.shape(ids_score)[0], 1], 0.0, 1.0)
      pdf     = scores - tf.expand_dims(tf.reduce_max(scores, axis=-1), axis=-1)
      pdf     = tf.nn.softmax(pdf, axis=-1)
      cdf     = tf.cumsum(pdf, axis=-1, exclusive=True)
      offset  = tf.where(cdf <= sample, tf.zeros_like(cdf), -2*tf.ones_like(cdf))
      sample  = cdf + offset
      action  = tf.argmax(sample, axis=-1, output_type=tf.int32, name=name)
      a_ucb   = None

      # # Sample via Gumbell distribution and no softmax
      # uniform = tf.random_uniform(tf.shape(scores), 0.0, 1.0)
      # gumbell = -tf.log(-tf.log(uniform))
      # sample  = scores + gumbell
      # action  = tf.argmax(sample, axis=-1, output_type=tf.int32, name=name)
      # a_ucb   = tf.argmax(mean + self.n_stds * std + gumbell, axis=-1, output_type=tf.int32)

      a_det   = tf.argmin(ids_score, axis=-1, output_type=tf.int32)

      a_diff_ds = tf.reduce_mean(tf.cast(tf.equal(a_det, action), tf.float32))
      tf.summary.scalar("debug/a_det_vs_stoch", a_diff_ds)

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
    # return self._act_eval_vote(agent_net, name)
    return self._act_eval_greedy(agent_net, name)


  def _restore(self, graph):
    super()._restore(graph)

    # Restore plot_train
    mean  = graph.get_tensor_by_name("plot/train/mean:0")
    std   = graph.get_tensor_by_name("plot/train/std:0")
    ids   = graph.get_tensor_by_name("plot/train/ids:0")
    a     = graph.get_tensor_by_name("plot/train/a:0")

    self.plot_train["train_actions"] = {
      "a_mean": dict(height=mean, a=a),
      "a_std":  dict(height=std,  a=a),
      "a_ids":  dict(height=ids,  a=a),
    }

    # Restore plot_eval
    self._restore_act_eval_plot(graph)
