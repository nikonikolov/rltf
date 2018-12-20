import numpy      as np
import tensorflow as tf


class ProbabilityDistribution:
  """Probability distribution class"""

  def sample(self):
    raise NotImplementedError()

  def logp(self, x):
    """Compute the log probability of x
    Args:
      x: tf.Tensor. Shape varies by class
    Returns:
      tf.Tensor of shape `[None]` and containing the log probabilities. Size is determined by batch_size
    """
    raise NotImplementedError()

  def entropy(self):
    """Compute the entropy H(p) = - E_p [log p], where p = self
    Returns:
      tf.Tensor of shape `[None]`. The size is determined by the batch size
    """
    raise NotImplementedError()

  def kl(self, other):
    """Compute D_KL(self || other)
    Returns:
      tf.Tensor of shape `[None]`. The size is determined by the batch size
    """
    raise NotImplementedError()



class CategoricalPD(ProbabilityDistribution):
  """Categorical distribution class"""

  def __init__(self, logits):
    """
    Args:
      logits: tf.Tensor, shape=[batch_size, n_classes]. Logits for the class probabilities
    """
    assert logits.shape.ndims == 2
    self.logits = logits


  def sample(self):
    """
    Returns:
      tf.Tensor of shape `[batch_size]`. Contains the indices of the sampled classes
    """
    return tf.squeeze(tf.multinomial(self.logits, num_samples=1), axis=-1)


  def logp(self, x):
    """
    Args:
      x: tf.Tensor, shape=`[batch_size]`. The class for which to compute logp
    Returns:
      tf.Tensor of shape `[batch_size]`
    """
    assert x.shape.ndims == 1
    if x.dtype.base_dtype == tf.uint8:
      x = tf.cast(x, dtype=tf.int32)
    #pylint: disable=invalid-unary-operand-type
    return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x, logits=self.logits)


  def entropy(self):
    logits  = self.logits - tf.stop_gradient(tf.reduce_max(self.logits, axis=-1, keepdims=True))
    exp     = tf.exp(logits)
    Z       = tf.reduce_sum(exp, axis=-1, keepdims=True)
    p       = exp / Z
    entropy = tf.reduce_sum(p * (tf.log(Z) - logits), axis=-1)
    return entropy


  def kl(self, other):
    assert isinstance(other, self.__class__)
    assert self.logits.shape[-1] == other.logits.shape[-1]

    plogits = self.logits  - tf.stop_gradient(tf.reduce_max(self.logits,  axis=-1, keepdims=True))
    qlogits = other.logits - tf.stop_gradient(tf.reduce_max(other.logits, axis=-1, keepdims=True))
    pexp    = tf.exp(plogits)
    qexp    = tf.exp(qlogits)
    pZ      = tf.reduce_sum(pexp, axis=-1, keepdims=True)
    qZ      = tf.reduce_sum(qexp, axis=-1, keepdims=True)
    pp      = pexp / pZ
    kl      = tf.reduce_sum(pp * ((plogits - tf.log(pZ)) - (qlogits - tf.log(qZ))), axis=-1)
    return kl



class DiagGaussianPD(ProbabilityDistribution):
  """Multivariate Gaussian distribution with diagonal covariance matrix"""

  def __init__(self, mean, logstd):

    assert mean.shape.ndims   == 2
    assert logstd.shape.ndims == 2

    self.mean   = mean                # [batch_size, self.dim]
    self.logstd = logstd              # [batch_size, self.dim] or [1, self.dim]
    self.std    = tf.exp(self.logstd)
    self.dim    = self.mean.shape.as_list()[1]


  def sample(self):
    """
    Returns:
      tf.Tensor of shape as `[None, self.dim]`. The size of the first dimension is
        determined from self.mean
    """
    return self.mean + tf.random_normal(shape=tf.shape(self.mean)) * self.std


  def logp(self, x):
    """
    Args:
      x: tf.Tensor, shape=`[batch_size, self.dim]`
    Returns:
      tf.Tensor of shape `[batch_size]`
    """
    assert x.shape.ndims == self.mean.shape.ndims
    logp = - 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
           - 0.5 * np.log(2.0 * np.pi) * self.dim - tf.reduce_sum(self.logstd, axis=-1)
    return logp


  def entropy(self):
    return 0.5 * self.dim * (np.log(2*np.pi) + 1) + tf.reduce_sum(self.logstd, axis=-1)


  def kl(self, other):
    assert isinstance(other, self.__class__)
    assert other.dim == self.dim

    return tf.reduce_sum( (
                            0.5 * tf.square((self.std + (self.mean - other.mean)) / other.std)
                            + other.logstd - self.logstd
                          ),
                          axis=-1) - 0.5 * self.dim
    # return tf.reduce_sum( (
    #                         0.5 * tf.square(tf.exp(self.logstd - other.std))
    #                         + 0.5 * tf.square((self.mean - other.mean) / other.std)
    #                         # + 0.5 * tf.square((self.mean - other.mean) / other.std)
    #                         + other.logstd - self.logstd
    #                       ),
    #                       axis=-1) - 0.5 * self.dim


  @property
  def dimension(self):
    return self.dim
