import tensorflow as tf

from rltf.models.dqn  import DQN


class DDQN(DQN):

  def _select_target(self, target_net):
    """Select the Double DQN target
    Args:
      target_net: `tf.Tensor`, shape `[None, n_actions]. The output from `self._nn_model()` for the target
    Returns:
      `tf.Tensor` of shape `[None]`
    """
    # Compute the Q-estimate with the agent network variables and select the maximizing action
    agent_net   = self._nn_model(self.obs_tp1, scope="agent_net")
    target_act  = tf.argmax(agent_net, axis=-1, output_type=tf.int32)

    # Select the target Q-function
    target_mask = tf.one_hot(target_act, self.n_actions, dtype=tf.float32)
    target_q    = tf.reduce_sum(target_net * target_mask, axis=-1)

    return target_q
