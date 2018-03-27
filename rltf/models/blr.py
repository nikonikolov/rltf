import tensorflow as tf

from rltf.models.tf_utils import cholesky_inverse
from rltf.models.tf_utils import sherman_morrison_inverse


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


  def weight_posterior(self, X, y, sequential=False):
    """Compute the weight posteriror of Bayesian Linear Regression
    Args:
      X: tf.Tensor, `shape=[None, D]`. The feature matrix
      y: tf.Tensor, `shape=[None, 1]`. The correct outputs
      sequential: bool. If True, updates are performed sequentially with one data point
        at a time. If False, update is done for the whole batch at once
    Returns:
      tf.Op which performs the update operation
    """
    X = self._add_bias(X)

    if sequential:
      return self._sequential_weight_posterior(X, y)
    return self._batch_weight_posterior(X, y)


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


  def _batch_weight_posterior(self, X, y):
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
    # w_Sigma = cholesky_inverse(w_Lambda)
    w_Sigma = tf.matrix_inverse(w_Lambda)

    # Compute the posterior mean
    w_mu = self._posterior_mean(X, y, self.w_mu, self.w_Lambda, w_Sigma)

    return self._tf_update_params(w_mu, w_Sigma, w_Lambda)


  def _sequential_weight_posterior(self, X, y):
    """Update the weight posterior point by point
    Args:
      X: tf.Tensor, `shape=[None, D]`. The feature matrix
      y: tf.Tensor, `shape=[None, 1]`. The correct outputs
    Returns:
      tf.Op which performs the update operation
    """
    N = tf.shape(X)[0]
    i = tf.constant(0)
    w_Sigma   = tf.identity(self.w_Sigma)
    w_Lambda  = tf.identity(self.w_Lambda)

    def body(i, w_Sigma, w_Lambda):
      x = tf.expand_dims(X[i], axis=0)
      w_Sigma, w_Lambda = self._posterior_cov_single_point(x, w_Sigma, w_Lambda)
      return i+1, w_Sigma, w_Lambda

    def cond(i, w_Sigma, w_Lambda):
      return tf.less(i, N)

    # Compute the posterior precision and covariance
    out = tf.while_loop(cond, body, [i, w_Sigma, w_Lambda], parallel_iterations=1, back_prop=False)
    _, w_Sigma, w_Lambda = out

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


  def _posterior_cov_single_point(self, x, w_prior_Sigma, w_prior_Lambda):
    """Compute the weight posteriror covariance and precision of
    Bayesian Linear Regression with only one new training data point
    Args:
      x: tf.Tensor, `shape=[1, D]`. The feature vector
      w_prior_Sigma: tf.Tensor, `shape=[D, D]`. The covariance of the Gaussian prior
      w_prior_Lambda: tf.Tensor, `shape=[D, D]`. The precision of the Gaussian prior
    Returns:
      w_Sigma: tf.Tensor, `shape=[D, D]`. The covariance of the Gaussian posterior
      w_Lambda: tf.Tensor, `shape=[D, D]`. The precision of the Gaussian posterior
    """

    x_T = tf.transpose(x)

    # Compute the posterior precision matrix
    w_Lambda = w_prior_Lambda + self.beta * tf.matmul(x_T, x)

    # Compute the posterior covariance matrix
    u = 1.0 / self.sigma * x_T
    w_Sigma = sherman_morrison_inverse(w_prior_Sigma, u, u)

    return w_Sigma, w_Lambda


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
