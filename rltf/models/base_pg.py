import gym
import tensorflow as tf

from rltf.models    import Model
from rltf.tf_utils  import tf_dist


class BasePG(Model):
  """Abstract Policy Gradients class"""

  def __init__(self, obs_shape, act_space, pi_opt_conf, vf_opt_conf, layers, activation, obs_norm,
               nn_std=False):
    """
    Args:
      obs_shape: list. Shape of the observation tensor
      act_space: gym.Space. Environment action space
      pi_opt_conf: rltf.optimizers.OptimizerConf. Configuration for the policy optimizer
      vf_opt_conf: rltf.optimizers.OptimizerConf. Configuration for the value function optimizer
      layers: list of ints or tuples. Contains the hidden layer spec for the neural net for both the
        policy and the value function. Each entry must be either a tuple of `(filters, size, stride)`
        for a convolutional layer or an `int` for a dense layer
      activation: lambda. Non-linear activation function for the hidden layers
      obs_norm: bool. Whether to normalize input observations
      nn_std: bool. If True, the standard deviation for a Gaussian policy is output of the neural net.
        Otherwise, a single trainable variable is used. Ignored for discrete action spaces
    """

    assert len(obs_shape) == 3 or len(obs_shape) == 1

    super().__init__()

    self.discrete     = isinstance(act_space, gym.spaces.Discrete)
    self.nn_std       = nn_std
    self.pi_opt_conf  = pi_opt_conf
    self.vf_opt_conf  = vf_opt_conf
    self.obs_norm     = obs_norm

    # Determine observation specs
    self.obs_shape    = obs_shape
    self.obs_dtype    = tf.uint8 if len(obs_shape) == 3 else tf.float32

    # Determine action specs
    if self.discrete:
      self.act_dim    = act_space.n
      self.act_shape  = []
      self.act_dtype  = tf.uint8 if act_space.n <= 256 else tf.int32
    else:
      self.act_dim    = act_space.shape[0]
      self.act_shape  = [self.act_dim]
      self.act_dtype  = tf.float32

    # Parse and save the layer specifications
    layers            = self._parse_layer_spec(layers)
    self.cnn_layers   = layers[0]
    self.dense_layers = layers[1]
    self.activation   = activation

    # General input TF placeholders
    self.obs_ph       = None
    self.act_ph       = None
    self.adv_ph       = None
    self.ret_ph       = None
    self.old_logp_ph  = None
    self.training     = None
    self.train_op     = None


  def _parse_layer_spec(self, layers):
    """Parse the layer spec provided in the constructor"""
    cnn_layers   = []
    dense_layers = []
    for layer in layers:
      if isinstance(layer, (tuple, list)):
        assert len(layer) == 3
        assert len(dense_layers) == 0, "All convolutional layers must come before any dense layer"
        cnn_layers.append(tuple(layer))
      elif isinstance(layer, int):
        dense_layers.append(layer)
      else:
        raise ValueError("Uknown layer specification {}. "
                         "Only dense and convlutional layers allowed".format(layer))
    return cnn_layers, dense_layers


  def _build_ph(self):
    """Build the input placehodlers"""
    self.obs_ph       = tf.placeholder(self.obs_dtype,  [None] + self.obs_shape, name="obs_ph")
    self.act_ph       = tf.placeholder(self.act_dtype,  [None] + self.act_shape, name="act_ph")
    self.adv_ph       = tf.placeholder(tf.float32,      [None],                  name="adv_ph")
    self.ret_ph       = tf.placeholder(tf.float32,      [None],                  name="ret_ph")
    self.old_logp_ph  = tf.placeholder(tf.float32,      [None],                  name="old_logp_ph")

    self.training     = tf.placeholder_with_default(True, (), name="training")


  def _build_nn(self, x, n_outputs):
    # Build the convolutional part of the network
    with tf.variable_scope("conv_net"):
      for filters, size, strides in self.cnn_layers:
        x = tf.layers.conv2d(x, filters=filters, kernel_size=size, strides=strides,
                             padding="SAME", activation=self.activation)

    if len(self.cnn_layers) > 0:
      x = tf.layers.flatten(x)

    # Build the dense part of the network
    with tf.variable_scope("dense_net"):
      for size in self.dense_layers:
        x = tf.layers.dense(x, units=size, activation=self.activation)
      x = tf.layers.dense(x, units=n_outputs, activation=None)

    return x


  def _pi_model(self, x, scope):
    """ Build policy network and the corresponding distribution
    Args:
      x: tf.Tensor. Tensor for the input
      scope: str. Scope in which all the model related variables should be created
    Returns:
      `ProbabilityDistribution`
    """

    # Determine the network output size
    if self.nn_std:
      n_outputs = 2*self.act_dim
    else:
      n_outputs = self.act_dim

    # Build the policy network
    with tf.variable_scope(scope):
      pi_out = self._build_nn(x, n_outputs)

      # Use Categorical distribution
      if self.discrete:
        pd = tf_dist.CategoricalPD(pi_out)

      # Use Gaussian distribution
      else:
        # Gaussian with logstd as network output
        if self.nn_std:
          mean, logstd = tf.split(pi_out, 2, axis=-1)
        # Gaussian with logstd as a single tf.Variable
        else:
          mean    = pi_out
          logstd  = tf.get_variable("logstd", shape=[1, n_outputs], initializer=tf.zeros_initializer())

        pd = tf_dist.DiagGaussianPD(mean, logstd)

    return pd


  def _vf_model(self, x, scope):
    """Build value function network"""
    with tf.variable_scope(scope):
      vf = self._build_nn(x, 1)
      vf = tf.squeeze(vf, axis=-1)

    self.ops_dict["vf"] = vf

    return vf


  @property
  def adv_norm(self):
    mean, var = tf.nn.moments(self.adv_ph, axes=[0], keep_dims=True)
    return (self.adv_ph - mean) / (tf.sqrt(var) + 1e-8)


  def _act_train(self, pi, vf, name):
    action  = tf.identity(pi.sample(), name=name)
    logp    = pi.log_prob(action)
    return dict(action=action, vf=vf, logp=logp)


  def _act_eval(self, pi, name):
    # If the policy is Categorical, we want the action with highest probability
    if self.discrete:
      # action = tf.identity(self.train_dict["action"], name=name)
      action = tf.identity(tf.argmax(pi.logits, axis=-1), name=name)
    # If the policy is Gaussian, we want the mean action during evaluation
    else:
      action = tf.identity(pi.mean, name=name)

    return dict(action=action)


  def _build_train_op(self, loss, pi_vars, vf_vars, name=None):
    pi_opt    = self.pi_opt_conf.build()
    vf_opt    = self.vf_opt_conf.build()
    train_pi  = pi_opt.minimize(loss, var_list=pi_vars)
    train_vf  = vf_opt.minimize(loss, var_list=vf_vars)
    train_op  = tf.group(train_pi, train_vf, name=name)

    self.ops_dict["train_pi"] = train_pi
    self.ops_dict["train_vf"] = train_vf

    return train_op


  def initialize(self, sess):
    pass


  def reset(self, sess):
    pass


  def action_train_ops(self, sess, state, run_dict=None):
    feed_dict = {self.obs_ph: state[None,:], self.training: False}
    return super()._action_train_ops(sess, run_dict, feed_dict=feed_dict)


  def action_eval_ops(self, sess, state, run_dict=None):
    feed_dict = {self.obs_ph: state[None,:], self.training: False}
    return super()._action_eval_ops(sess, run_dict, feed_dict=feed_dict)
