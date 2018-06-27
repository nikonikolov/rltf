import tensorflow as tf

from rltf.models.tf_utils import woodburry_inverse


class BLR(tf.layers.Layer):

  def __init__(self, tau, sigma_e, mode="mean", w_dim=None, bias=False, dtype=tf.float64, name=None):

    super().__init__(trainable=False, dtype=dtype, name=name)

    assert mode in ["mean", "ts"]

    self.sigma  = sigma_e
    self.beta   = 1.0 / self.sigma**2
    self.tau    = tau
    self.bias   = bias
    self.w_dim  = w_dim
    self.mode   = mode

    # Custom TF Tensors and Ops
    self.w_mu     = None
    self.w_Sigma  = None
    self.w_Lambda = None
    self.w        = None  # Sampled value for w when using Thompson Sampling
    self.reset_op = None  # Reset all trainable variables to their inNeed to make sure it is itial values

    self.input_spec = tf.layers.InputSpec(min_ndim=2, max_ndim=2)


  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)

    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `BLR` should be defined. Found `None`.')

    if self.w_dim is None:
      self.w_dim      = input_shape[-1].value
      self.input_spec = tf.layers.InputSpec(min_ndim=2, max_ndim=2, axes={-1: self.w_dim})
    else:
      assert self.w_dim == input_shape[-1].value

    I = tf.eye(self.w_dim, dtype=self.dtype)

    mu_init     = tf.zeros([self.w_dim, 1], dtype=self.dtype)
    Sigma_init  = 1.0/self.tau * I
    Lambda_init = self.tau * I

    self.w_mu     = self.add_variable("w_mu",
                                      shape=[self.w_dim, 1],
                                      # initializer=lambda *args, **kwargs: mu_init,
                                      initializer=tf.zeros_initializer,
                                      trainable=True)

    self.w_Sigma  = self.add_variable("w_Sigma",
                                      shape=[self.w_dim, self.w_dim],
                                      initializer=lambda *args, **kwargs: Sigma_init,
                                      trainable=True)

    self.w_Lambda = self.add_variable("w_Lambda",
                                      shape=[self.w_dim, self.w_dim],
                                      initializer=lambda *args, **kwargs: Lambda_init,
                                      trainable=True)

    self.w        = self.add_variable("w",
                                      shape=[self.w_dim, 1],
                                      # initializer=lambda *args, **kwargs: mu_init,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)

    # Build the reset op
    self.reset_op = self._tf_update_params(mu_init, Sigma_init, Lambda_init)

    # Add debug histogrrams
    # tf.summary.histogram("debug/BLR/w_mu",      self.w_mu)
    # tf.summary.histogram("debug/BLR/w_Sigma",   self.w_Sigma)
    # tf.summary.histogram("debug/BLR/w_Lambda",  self.w_Lambda)
    # tf.summary.histogram("debug/BLR/w_ts",      self.w)

    self.built = True


  def call(self, inputs, **kwargs):
    """ Compute the posterior predictive distribution
    Args:
      X: tf.Tensor, `shape=[None, D]`. The feature matrix
    Returns:
      List of `tf.Tensor`s:
        mu: tf.Tensor, `shape=[None, 1]. The mean at each test point
        var: tf.Tensor, `shape=[None, 1]. The variance at each test point
    """
    X = self._cast_input(inputs)

    # Thompson Sampling Output
    if self.mode == "ts":
      mu = tf.matmul(X, self.w)
    # Bayesian Regression Output
    else:
      mu = tf.matmul(X, self.w_mu)

    # var ends up being diag(sigma**2 + matmul(matmul(X, w_Sigma), X.T))
    var = self.sigma**2 + tf.reduce_sum(tf.matmul(X, self.w_Sigma) * X, axis=-1, keepdims=True)
    # std = tf.sqrt(var)

    outputs = [mu, var]
    outputs = [self._cast_output(t) for t in outputs]
    return outputs


  def train(self, X, y):
    """Compute the weight posteriror of Bayesian Linear Regression
    Args:
      X: tf.Tensor, `shape=[None, D]`. The feature matrix
      y: tf.Tensor, `shape=[None, 1]`. The correct outputs
    Returns:
      tf.Op which performs the update operation
    """
    X = self._cast_input(X)
    y = self._cast_input(y)

    # Compute the posterior precision matrix
    w_Lambda = self.w_Lambda + self.beta * tf.matmul(X, X, transpose_a=True)

    # Compute the posterior covariance matrix
    X_norm  = 1.0 / self.sigma * X
    w_Sigma = woodburry_inverse(self.w_Sigma, tf.transpose(X_norm), X_norm)

    error = tf.losses.mean_squared_error(tf.matmul(w_Lambda, w_Sigma), tf.eye(self.w_dim))
    tf.summary.scalar("debug/BLR/inv_error", error)

    # Compute the posterior mean
    w_mu = tf.matmul(w_Sigma, self.beta * tf.matmul(X, y, True) + tf.matmul(self.w_Lambda, self.w_mu))

    return self._tf_update_params(w_mu, w_Sigma, w_Lambda)


  def resample_w(self, cholesky=False):
    sample = tf.random_normal(shape=self.w_mu.shape, dtype=self.dtype)

    # Compute A s.t. A A^T = w_Sigma. Note that SVD and Cholesky give different A
    if cholesky:
      # Use cholesky
      A = tf.cholesky(self.w_Sigma)
    else:
      # Use SVD
      S, U, _ = tf.svd(self.w_Sigma)
      A = tf.matmul(U, tf.diag(tf.sqrt(S)))

    w = self.w_mu + tf.matmul(A, sample)
    return tf.assign(self.w, w, name="resample_w")


  @property
  def reset(self):
    return self.reset_op


  @property
  def trainable_weights(self):
    return self._trainable_weights or []


  def _tf_update_params(self, w_mu, w_Sigma, w_Lambda):
    """
    Returns:
      tf.Op which performs an update on all weight parameters
    """
    mu_op     = tf.assign(self.w_mu,      w_mu)
    Sigma_op  = tf.assign(self.w_Sigma,   w_Sigma)
    Lambda_op = tf.assign(self.w_Lambda,  w_Lambda)
    return tf.group(mu_op, Sigma_op, Lambda_op)


  def _cast_input(self, x):
    if self.dtype == tf.float64 and x.dtype.base_dtype != tf.float64:
      x = tf.cast(x, self.dtype)
    return x


  def _cast_output(self, x):
    if x.dtype.base_dtype != tf.float32:
      x = tf.cast(x, tf.float32)
    return x
