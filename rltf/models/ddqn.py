import tensorflow as tf

from rltf.models  import DQN


class DDQN(DQN):

  def _compute_target(self, agent_net, target_net):
    done_mask   = tf.cast(tf.logical_not(self._done_ph), tf.float32)
    target_mask = tf.argmax(agent_net, axis=-1, output_type=tf.int32)
    target_mask = tf.one_hot(target_mask, self.n_actions, on_value=True, off_value=False, dtype=tf.bool)
    target_q    = tf.boolean_mask(target_net, target_mask)
    target_q    = self.rew_t_ph + self.gamma * done_mask * target_q

    return target_q
