import numpy      as np
import tensorflow as tf

from rltf.models.model  import Model
from rltf.models        import tf_utils


def dqn_nn(img_in, n_actions, scope):
  """ Build the DQN architecture - as described in the original paper

  Args:
    img_in: tf.Tensor. Tensor for the input image
    n_actions: int. Number of possible output actions
    scope: str. Scope in which all the model related variables should be created

  Returns:
    `tf.Tensor` of shape `[batch_size, n_actions]`. Contains the Q-function for each action
  """
  with tf.variable_scope(scope, reuse=False):
    x = img_in
    with tf.variable_scope("convnet"):
      # original architecture
      x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    with tf.variable_scope("action_value"):
      x = tf.layers.dense(x, units=512,       activation=tf.nn.relu)
      x = tf.layers.dense(x, units=n_actions, activation=None)
    return x


class DQN(Model):
  """ Class for the DQN model """

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, huber_loss=True):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
      huber_loss: bool. Whether to use huber loss or not
    """
    
    super().__init__()

    self.gamma      = gamma
    self.opt_conf   = opt_conf
    self.huber_loss = huber_loss
    self.nn_model   = dqn_nn

    self.obs_shape  = obs_shape
    self.obs_dtype  = tf.uint8
    self.n_actions  = n_actions
    self.act_shape  = []
    self.act_dtype  = tf.uint8

  def build(self):

    super()._build()
    # Input placeholders
    # self._obs_t_ph    = tf.placeholder(tf.uint8,   [None] + self.obs_shape, name="obs_t_ph")
    # self._act_t_ph    = tf.placeholder(tf.uint8,   [None],                  name="act_t_ph")
    # self._rew_t_ph    = tf.placeholder(tf.float32, [None],                  name="rew_t_ph")
    # self._obs_tp1_ph  = tf.placeholder(tf.uint8,   [None] + self.obs_shape, name="obs_tp1_ph")
    # self._done_ph     = tf.placeholder(tf.bool,    [None],                  name="done_ph")

    # In this case, casting on GPU ensures lower data transfer times
    obs_t_float   = tf.cast(self._obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(self._obs_tp1_ph, tf.float32) / 255.0
    act_t         = tf.cast(self._act_t_ph,   tf.int32)

    # Construct the Q-network and the target network 
    q           = self.nn_model(obs_t_float,   self.n_actions, scope="agent_net")
    target_q    = self.nn_model(obs_tp1_float, self.n_actions, scope="target_net")
      
    # Get the Q value for the played action
    act_mask    = tf.one_hot(act_t, self.n_actions, on_value=True, off_value=False, dtype=tf.bool)
    sample_q    = tf.boolean_mask(q, act_mask)
    
    # Get the target Q value
    done_mask   = tf.cast(tf.logical_not(self._done_ph), tf.float32)
    target_q    = tf.reduce_max(target_q, axis=-1)
    target_q    = self.rew_t_ph + self.gamma * done_mask * target_q

    # Compute the loss
    loss_fn     = tf.losses.huber_loss if self.huber_loss else tf.losses.mean_squared_error
    loss        = loss_fn(target_q, sample_q)

    agent_vars  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_net')
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

    # Build the optimizer
    optimizer   = self.opt_conf.build()
    # Create the training Op
    self._train_op = optimizer.minimize(loss, var_list=agent_vars, name="train_op")

    # Create the Op to update the target
    self._update_target = tf_utils.assign_values(target_vars, agent_vars, name="update_target")

    # Save the Q-function estimate tensor
    self._q   = tf.identity(q, name="q_fn")

    # Add summaries
    tf.summary.scalar("loss", loss)


  def _restore(self, graph):
    # Get Q-function Tensor
    self._q = graph.get_tensor_by_name("q_fn:0")


  def initialize(self, sess):
    """Initialize the model. See Model.initialize()"""
    sess.run(self._update_target)


  def control_action(self, sess, state):
    q_vals  = sess.run(self._q, feed_dict={self.obs_t_ph: state[None,:]})
    action  = np.argmax(q_vals)
    return action

