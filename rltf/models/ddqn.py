import tensorflow as tf

from rltf.models  import DQN


class DDQN(DQN):

  def _compute_target(self, target_net):
    # Compute the Q-estimate with the agent network variables
    agent_net   = self._nn_model(self._obs_tp1, scope="agent_net")

    # Compute the target action
    target_mask = tf.argmax(agent_net, axis=-1, output_type=tf.int32)
    target_mask = tf.one_hot(target_mask, self.n_actions, dtype=tf.float32)

    # Compute the target
    done_mask   = tf.cast(tf.logical_not(self._done_ph), tf.float32)
    target_q    = tf.reduce_sum(target_net * target_mask, axis=-1)
    target_q    = self.rew_t_ph + self.gamma * done_mask * target_q
    target_q    = tf.stop_gradient(target_q)

    return target_q
