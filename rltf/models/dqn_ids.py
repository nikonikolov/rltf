import tensorflow as tf

from rltf.models.bstrap_dqn import BaseBstrapDQN


class DQN_IDS(BaseBstrapDQN):
  """IDS policy from Boostrapped DQN"""

  def __init__(self, n_stds, **kwargs):
    super().__init__(**kwargs)

    self.n_stds = n_stds    # Number of standard deviations for computing uncertainty
    self.rho2   = 1.0**2    # Return distribution variance


  def _act_train(self, agent_net, name):
    mean      = tf.reduce_mean(agent_net, axis=1)
    zero_mean = agent_net - tf.expand_dims(mean, axis=-2)
    var       = tf.reduce_mean(tf.square(zero_mean), axis=1)
    std       = tf.sqrt(var)
    regret    = tf.reduce_max(mean + self.n_stds * std, axis=-1, keepdims=True)
    regret    = regret - (mean - self.n_stds * std)
    regret_sq = tf.square(regret)
    info_gain = tf.log(1 + var / self.rho2) + 1e-5
    ids_score = regret_sq / info_gain
    action    = tf.argmin(ids_score, axis=-1, output_type=tf.int32, name=name)

    # Add debug histograms
    tf.summary.histogram("debug/a_mean",    mean)
    tf.summary.histogram("debug/a_std",     std)
    tf.summary.histogram("debug/a_regret",  regret)
    tf.summary.histogram("debug/a_info",    info_gain)
    tf.summary.histogram("debug/a_ids",     ids_score)

    # Set the plottable tensors for episode recordings
    p_a     = tf.identity(action[0],    name="plot/train/a")
    p_mean  = tf.identity(mean[0],      name="plot/train/mean")
    p_std   = tf.identity(std[0],       name="plot/train/std")
    p_ids   = tf.identity(ids_score[0], name="plot/train/ids")

    train_actions = {
      "a_mean": dict(height=p_mean, a=p_a),
      "a_std":  dict(height=p_std,  a=p_a),
      "a_ids":  dict(height=p_ids,  a=p_a),
    }
    self.plot_conf.set_train_spec(dict(train_actions=train_actions))

    return dict(action=action)


  def _act_eval(self, agent_net, name):
    return dict(action=self._act_eval_greedy(agent_net, name))
