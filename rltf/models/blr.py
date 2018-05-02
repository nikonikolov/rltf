import tensorflow as tf

from rltf.models.tf_utils import woodburry_inverse


class BLR:
  """Bayesian Linear Regression implemented in TF"""

  def __init__(self, sigma, tau, w_dim, auto_bias=True, dtype=tf.float64):
    """
    Args:
      sigma: float. Standard deviation of observation noise
      tau: float. Standard deviation of parameter prior
      w_dim: int. The number of input features. If `auto_bias=True`, then this
        number will be automatically incremented by 1
      auto_bias: bool. If True, a bias feature will automatically be added to
        all input data.
      woodburry: bool. If True, use Woodburry formula to compute the covariance matrix
        as inverse of the precision. Else use `tf.inverse()`
    """

    self.auto_bias = auto_bias

    self.sigma  = sigma
    self.beta   = 1.0 / self.sigma**2
    self.tau    = tau
    self.w_dim  = w_dim if not auto_bias else w_dim+1
    self.dtype  = dtype

    # Custom TF Tensors and Ops
    self.w_mu     = None
    self.w_Sigma  = None
    self.w_Lambda = None
    self.w_ts     = None  # Sampled value for w when using Thompson Sampling


  def build(self):
    I     = tf.eye(self.w_dim, dtype=self.dtype)
    tau2  = self.tau**2

    mu_init     = tf.zeros([self.w_dim, 1], dtype=self.dtype)
    Sigma_init  = tau2 * I
    Lambda_init = 1.0/tau2 * I

    # Variables for the parameters of the weight distribution in BLR
    self.w_mu     = tf.Variable(mu_init,      dtype=self.dtype, trainable=False)
    self.w_Sigma  = tf.Variable(Sigma_init,   dtype=self.dtype, trainable=False)
    self.w_Lambda = tf.Variable(Lambda_init,  dtype=self.dtype, trainable=False)
    self.w_ts     = tf.Variable(mu_init,      dtype=self.dtype, trainable=False)

    # self.w_mu     = tf.get_variable("w_mu",     initializer=mu_init,      dtype=self.dtype, trainable=False)
    # self.w_Sigma  = tf.get_variable("w_Sigma",  initializer=Sigma_init,   dtype=self.dtype, trainable=False)
    # self.w_Lambda = tf.get_variable("w_Lambda", initializer=Lambda_init,  dtype=self.dtype, trainable=False)
    # self.w_ts     = tf.get_variable("w_ts",     initializer=mu_init,      dtype=self.dtype, trainable=False)

    tf.summary.histogram("debug/BLR_w_mu",      self.w_mu)
    tf.summary.histogram("debug/BLR_w_Sigma",   self.w_Sigma)
    tf.summary.histogram("debug/BLR_w_Lambda",  self.w_Lambda)
    tf.summary.histogram("debug/BLR_w_ts",      self.w_ts)

    # Build the reset op
    self.reset    = self._tf_update_params(mu_init, Sigma_init, Lambda_init)


  def weight_posterior(self, X, y):
    """Compute the weight posteriror of Bayesian Linear Regression
    Args:
      X: tf.Tensor, `shape=[None, D]`. The feature matrix
      y: tf.Tensor, `shape=[None, 1]`. The correct outputs
    Returns:
      tf.Op which performs the update operation
    """
    X = self._cast_input(X)
    y = self._cast_input(y)
    X = self._add_bias(X)

    return self._weight_posterior(X, y)


  def _add_bias(self, x):
    """Add bias to the feature
    Args:
      x: tf.Tensor, `shape=[None, D]`. The feature matrix
    """
    if self.auto_bias:
      bias  = tf.ones(shape=[tf.shape(x)[0], 1], dtype=self.dtype)
      x     = tf.concat([x, bias], axis=-1)
    return x


  def _cast_input(self, x):
    if self.dtype == tf.float64 and x.dtype.base_dtype != tf.float64:
      x = tf.cast(x, self.dtype)
    return x


  def _cast_output(self, x):
    if x.dtype.base_dtype != tf.float32:
      x = tf.cast(x, tf.float32)
    return x


  def _tf_update_params(self, w_mu, w_Sigma, w_Lambda):
    """
    Returns:
      tf.Op which performs an update on all weight parameters
    """
    # t     = tf.Variable(0, dtype=tf.int32)
    # t     = tf.assign(t, t+1)
    # I     = tf.cast(tf.matmul(w_Sigma, w_Lambda), tf.float32)
    # error = tf.losses.mean_squared_error(I, tf.eye(self.w_dim, dtype=tf.float32)) * self.w_dim
    # w_mu  = tf.Print(w_mu, [error, t], "error inverse (step): ")

    mu_op     = tf.assign(self.w_mu,      w_mu)
    Sigma_op  = tf.assign(self.w_Sigma,   w_Sigma)
    Lambda_op = tf.assign(self.w_Lambda,  w_Lambda)

    return tf.group(mu_op, Sigma_op, Lambda_op)


  def _weight_posterior(self, X, y):
    """Compute the weight posteriror of Bayesian Linear Regression
    Args:
      X: tf.Tensor, `shape=[None, D]`. The feature matrix
      y: tf.Tensor, `shape=[None, 1]`. The correct outputs
    Returns:
      tf.Op which performs the update operation
    """
    # Compute the posterior precision matrix
    w_Lambda = self.w_Lambda + self.beta * tf.matmul(X, X, transpose_a=True)

    # Compute the posterior covariance matrix
    X_norm  = 1.0 / self.sigma * X
    w_Sigma = woodburry_inverse(self.w_Sigma, tf.transpose(X_norm), X_norm)

    error = tf.losses.mean_squared_error(tf.matmul(w_Lambda, w_Sigma), tf.eye(self.w_dim))
    tf.summary.scalar("debug/BLR_inv_error", error)

    # Compute the posterior mean
    w_mu = self._posterior_mean(X, y, self.w_mu, self.w_Lambda, w_Sigma)

    return self._tf_update_params(w_mu, w_Sigma, w_Lambda)


  def _posterior_mean(self, X, y, w_prior_mu, w_prior_Lambda, w_Sigma):
    """Given the weight posterior covariance, compute the weight posteriror mean
    Args:
      X: tf.Tensor, `shape=[None, D]`. The feature vector
      y: tf.Tensor, `shape=[None]`. The correct labels for the data
      w_prior_mu: tf.Tensor, `shape=[D, 1]`. The mean of the Gaussian prior
      w_prior_Lambda: tf.Tensor, `shape=[D, D]`. The precision of the Gaussian prior
    Returns:
      w_mu: tf.Tensor, `shape=[D, 1]`. Mean of the Gaussian posterior
    """
    w_mu = tf.matmul(w_Sigma, self.beta * tf.matmul(X, y, True) + tf.matmul(w_prior_Lambda, w_prior_mu))
    w_mu = tf.cond(tf.reduce_all(tf.equal(X, tf.zeros_like(X))), lambda: w_prior_mu, lambda: w_mu)
    return w_mu


  def predict(self, X):
    """ Compute the posterior predictive distribution
    Args:
      X: tf.Tensor, `shape=[None, D]`. The feature matrix
    Returns:
      mu: tf.Tensor, `shape=[None, 1]. The mean at each test point
      std: tf.Tensor, `shape=[None, 1]. The standard deviation at each test point
    """
    X   = self._cast_input(X)
    X   = self._add_bias(X)

    mu  = tf.matmul(X, self.w_mu)
    # var ends up being diag(sigma**2 + matmul(matmul(X, w_Sigma), X.T))
    var = self.sigma**2 + tf.reduce_sum(tf.matmul(X, self.w_Sigma) * X, axis=-1, keep_dims=True)
    std = tf.sqrt(var)

    mu  = self._cast_output(mu)
    std = self._cast_output(std)

    return mu, std


  def predict_thompson(self, X):
    """ Compute the predicted y using a sampled parameter w
    Args:
      X: tf.Tensor, `shape=[None, D]`. The feature matrix
    Returns:
      y: tf.Tensor, `shape=[None, 1]. The mean at each test point
    """
    X   = self._cast_input(X)
    X   = self._add_bias(X)
    y   = tf.matmul(X, self.w_ts)
    y   = self._cast_output(y)
    return y


  def reset_thompson(self, cholesky=False):
    sample = tf.random_normal(shape=self.w_mu.shape)

    # Compute A s.t. A A^T = w_Sigma
    if cholesky:
      # Use cholesky
      A = tf.cholesky(self.w_Sigma)
    else:
      # Use SVD
      S, U, _ = tf.svd(self.w_Sigma)
      A = tf.matmul(U, tf.diag(tf.sqrt(S)))

    return tf.assign(self.w_ts, self.w_mu + tf.matmul(A, sample))
