import tensorflow as tf

from rltf.models.bstrap_dqn import BaseBstrapDQN


class DQN_Ensemble(BaseBstrapDQN):
  """Ensemble policy from Boostrapped DQN"""

  def _act_train(self, agent_net, name):
    action = self._act_eval_vote(agent_net, name)
    # Set the plottable tensors for episode recordings
    self.plot_conf.set_train_spec(dict(eval_actions=self.plot_conf.true_eval_spec["eval_actions"]))
    return dict(action=action)


  def _act_eval(self, agent_net, name):
    return dict(action=tf.identity(self.train_dict["action"], name=name))
