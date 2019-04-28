import tensorflow as tf

from rltf.models.bstrap_dqn import BaseBstrapDQN


class DQN_UCB(BaseBstrapDQN):
  """UCB policy from Boostrapped DQN"""

  def __init__(self, n_stds=0.1, **kwargs):
    super().__init__(**kwargs)
    self.n_stds = n_stds       # Number of standard deviations for computing uncertainty


  def _act_train(self, agent_net, name):
    mean    = tf.reduce_mean(agent_net, axis=1)
    std     = agent_net - tf.expand_dims(mean, axis=-2)
    std     = tf.sqrt(tf.reduce_mean(tf.square(std), axis=1))
    action  = tf.argmax(mean + self.n_stds * std, axis=-1, output_type=tf.int32, name=name)

    # Add debug histograms
    tf.summary.histogram("debug/a_std",   std)
    tf.summary.histogram("debug/a_mean",  mean)

    return dict(action=action)


  def _act_eval(self, agent_net, name):
    return dict(action=self._act_eval_vote(agent_net, name))
