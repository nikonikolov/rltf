import tensorflow as tf

from rltf.models.tf_utils import woodburry_inverse


class BayesianLinearRegression:
  """Bayesian Linear Regression implemented in TF"""

  def __init__(self, sigma, tau, w_dim, auto_bias=True):
    """
    Args:
      sigma: float. Standard deviation of observation noise
      tau: float. Standard deviation of parameter prior
      w_dim: int. The number of input features. If `auto_bias=True`, then this
        number will be automatically incremented by 1
      auto_bias: bool. If True, a bias feature will automatically be added to
        all input data.
    """

    self.sigma  = sigma
    self.beta   = 1.0 / self.sigma**2
    self.tau    = tau
    self.w_dim  = w_dim

    self.auto_bias = auto_bias
    if self.auto_bias:
      self.w_dim += 1

    # Custom TF Tensors and Ops
    self.w_mu     = None
    self.w_Sigma  = None
    self.w_Lambda = None


  def build(self):

    I = tf.eye(self.w_dim)
    # Variables for the parameters of the weight distribution in BLR
    self.w_mu     = tf.Variable(tf.zeros([self.w_dim, 1]),  trainable=False)
    self.w_Sigma  = tf.Variable(    (self.tau**2) * I,      trainable=False)
    self.w_Lambda = tf.Variable(1.0/(self.tau**2) * I,      trainable=False)


  def weight_posterior(self, X, y):
    """Compute the weight posteriror of Bayesian Linear Regression
    Args:
      X: tf.Tensor, `shape=[None, D]`. The feature matrix
      y: tf.Tensor, `shape=[None, 1]`. The correct outputs
    Returns:
      tf.Op which performs the update operation
    """
    X = self._add_bias(X)

    return self._weight_posterior(X, y)


  def _add_bias(self, x):
    """Add bias to the feature
    Args:
      x: tf.Tensor, `shape=[None, D]`. The feature matrix
    """
    if self.auto_bias:
      bias  = tf.ones(shape=[tf.shape(x)[0], 1], dtype=tf.float32)
      x     = tf.concat([x, bias], axis=-1)
    return x


  def _tf_update_params(self, w_mu, w_Sigma, w_Lambda):
    """
    Returns:
      tf.Op which performs an update on all weight parameters
    """
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

    X_scaled = 1.0/self.sigma * X

    # Compute the posterior covariance matrix
    w_Sigma = woodburry_inverse(self.w_Sigma, tf.transpose(X_scaled), X_scaled)
    # w_Sigma = tf.matrix_inverse(w_Lambda)

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
    X   = self._add_bias(X)
    mu  = tf.matmul(X, self.w_mu)
    # var ends up being diag(sigma**2 + matmul(matmul(X, w_Sigma), X.T))
    var = self.sigma**2 + tf.reduce_sum(tf.matmul(X, self.w_Sigma) * X, axis=-1, keep_dims=True)
    std = tf.sqrt(var)

    return mu, std
