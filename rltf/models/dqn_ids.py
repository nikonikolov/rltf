import tensorflow as tf

from rltf.models  import DQN
from rltf.models  import BayesianLinearRegression
from rltf.models  import tf_utils


class DQN_IDS_BLR(DQN):

  def __init__(self, obs_shape, n_actions, opt_conf, gamma, sigma, tau, rho, huber_loss=True):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      n_actions: int. Number of possible actions
      opt_conf: rltf.optimizers.OptimizerConf. Configuration for the optimizer
      gamma: float. Discount factor
      sigma: float. Standard deviation of the noise observation for BLR
      tau: float. Standard deviation for the weight prior in BLR
      rho: float. Reward observation noise for IDS
      huber_loss: bool. Whether to use huber loss or not
    """

    super().__init__(obs_shape, n_actions, opt_conf, gamma, huber_loss)

    self.dim_phi  = 512
    self.rho      = rho
    self.n_stds   = 1.0     # Number of standard deviations for computing the regret bound

    blr_params = {"sigma": sigma, "tau": tau, "w_dim": self.dim_phi+1, "auto_bias": False}

    self.blr_models = [BayesianLinearRegression(**blr_params) for _ in range(self.n_actions)]

    # Custom TF Tensors and Ops
    # self.blr_train  = None


  def build(self):

    super()._build()

    # In this case, casting on GPU ensures lower data transfer times
    obs_t       = tf.cast(self._obs_t_ph,   tf.float32) / 255.0
    obs_tp1     = tf.cast(self._obs_tp1_ph, tf.float32) / 255.0

    # Construct the Q-network and the target network
    agent_net, phi  = self._nn_model(obs_t,   scope="agent_net")
    target_net, _   = self._nn_model(obs_tp1, scope="target_net")

    # Compute the estimated Q-function and its backup value
    estimate    = self._compute_estimate(agent_net)
    target      = self._compute_target(agent_net, target_net)

    # Compute the loss
    loss        = self._compute_loss(estimate, target)

    agent_vars  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_net')
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

    # Create the Op to update the target
    self._update_target = tf_utils.assign_vars(target_vars, agent_vars, name="update_target")

    # Build the optimizer
    optimizer   = self.opt_conf.build()
    net_train   = optimizer.minimize(loss, var_list=agent_vars, name="train_op")

    # Normalize features
    # phi = tf.layers.batch_normalization(phi, axis=-1, training=True, name="batch_norm1")

    # Add bias to the feature transformations
    bias    = tf.ones(shape=[tf.shape(phi)[0], 1], dtype=tf.float32)
    phi     = tf.concat([phi, bias], axis=-1)

    target  = tf.expand_dims(target, axis=-1)

    # Build the Bayesian Regerssion ops and estimates for all actions
    y_means, y_stds, w_updates = [], [], []
    for i in range(self.n_actions):
      self.blr_models[i].build()
      mask = tf.equal(self._act_t_ph, i)
      X = tf.boolean_mask(phi,    mask)
      y = tf.boolean_mask(target, mask)
      b = tf.shape(X)[0]
      X = tf.cond(tf.not_equal(b, 0), lambda: X, lambda: tf.zeros([1, self.dim_phi+1]))
      y = tf.cond(tf.not_equal(b, 0), lambda: y, lambda: tf.zeros([1, 1]))
      w_update_op = self.blr_models[i].weight_posterior(X, y)
      y_m, y_std  = self.blr_models[i].predict(phi)

      w_updates.append(w_update_op)
      y_stds.append(y_std)
      y_means.append(y_m)

    # Create weight update op and the train op
    blr_train       = tf.group(*w_updates, name="blr_train")
    self._train_op  = tf.group(net_train, blr_train, name="train_op")

    # Compute the IDS action
    act_means     = tf.concat(y_means, axis=-1)     # output shape [None, n_actions]
    act_stds      = tf.concat(y_stds,  axis=-1)     # output shape [None, n_actions]
    act_regret    = tf.reduce_max(act_means + self.n_stds * act_stds, axis=-1)
    act_regret    = act_regret - (act_means - self.n_stds * act_stds)
    act_regret_sq = tf.square(act_regret)
    act_info_gain = tf.log(1 + tf.square(act_stds) / self.rho**2)
    act_ids       = tf.div(act_regret_sq, act_info_gain)

    self.a_train  = tf.argmin(act_ids, axis=-1, output_type=tf.int32, name="a_train")
    self.a_eval   = self._act_eval(agent_net,  name="a_eval")

    # Add summaries
    tf.summary.scalar("loss", loss)


  def _nn_model(self, x, scope):
    """ Build the DQN architecture - as described in the original paper
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created

    Returns:
      `tf.Tensor` of shape `[batch_size, n_actions]`. Contains the Q-function for each action
    """
    n_actions = self.n_actions

    with tf.variable_scope(scope, reuse=False):
      with tf.variable_scope("convnet"):
        # original architecture
        x = tf.layers.conv2d(x, filters=32, kernel_size=8, strides=4, padding="SAME", activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding="SAME", activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
      x = tf.layers.flatten(x)
      with tf.variable_scope("action_value"):
        phi = tf.layers.dense(x, units=self.dim_phi, activation=tf.nn.relu)
        x = tf.layers.dense(phi, units=n_actions, activation=None)
      return x, phi


  # def _restore(self, graph):
  #   super()._restore()
  #   # (TODO): Restore BLR variables
  #   # self.blr_train  = graph.get_operation_by_name("blr_train")


  def _act_eval(self, agent_net, name):
    action = tf.argmax(agent_net, axis=-1, output_type=tf.int32, name=name)
    return action
