import numpy      as np
import tensorflow as tf


class MultivariateNormalDiag(tf.distributions.Distribution):
  """Multivariate Gaussian distribution with diagonal covariance matrix"""

  def __init__(self, loc, log_scale=None, scale=None, validate_args=False, allow_nan_stats=True):

    parameters = dict(locals())

    assert log_scale.shape.ndims == 2
    assert (log_scale is None) != (scale is None)

    with tf.name_scope(self.__class__.__name__):

      if log_scale is not None:
        loc       = tf.identity(loc, name="loc")
        scale     = tf.exp(log_scale, name="scale")
        log_scale = tf.identity(log_scale, name="log_scale")
      elif scale is not None:
        with tf.control_dependencies([tf.assert_positive(scale)]):
          loc       = tf.identity(loc, name="loc")
          scale     = tf.identity(scale, name="scale")
          log_scale = tf.log(scale, name="log_scale")

      assert loc.dtype.base_dtype == tf.float32 or loc.dtype.base_dtype == tf.float64
      assert loc.dtype.base_dtype == log_scale.dtype.base_dtype == scale.dtype.base_dtype

    self.loc        = loc             # [batch_size, self.dim]
    self.log_scale  = log_scale       # [batch_size, self.dim] or [1, self.dim]
    self.scale      = scale           # [batch_size, self.dim] or [1, self.dim]
    self.dim        = self.loc.shape.as_list()[1]

    super().__init__(dtype=self.loc.dtype,
                     reparameterization_type=tf.distributions.FULLY_REPARAMETERIZED,
                     validate_args=validate_args,
                     allow_nan_stats=allow_nan_stats,
                     parameters=parameters,
                    )


  def sample(self):
    """
    Returns:
      tf.Tensor of shape as `[None, self.dim]`. The size of the first dimension is
        determined from self.loc
    """
    return self.loc + tf.random_normal(shape=tf.shape(self.loc)) * self.scale


  def _log_prob(self, value):
    """
    Args:
      value: tf.Tensor, shape=`[batch_size, self.dim]`
    Returns:
      tf.Tensor of shape `[batch_size]`
    """
    assert value.shape.ndims == self.loc.shape.ndims
    logp = - 0.5 * tf.reduce_sum(tf.square((value - self.loc) / self.scale), axis=-1) \
           - 0.5 * np.log(2.0 * np.pi) * self.dim - tf.reduce_sum(self.log_scale, axis=-1)
    return logp


  def _entropy(self):
    return 0.5 * self.dim * (np.log(2*np.pi) + 1) + tf.reduce_sum(self.log_scale, axis=-1)


  def _kl_divergence(self, other):
    assert isinstance(other, self.__class__)
    assert other.dim == self.dim

    return tf.reduce_sum( (
                            0.5 * tf.square(self.scale / other.scale) +
                            0.5 * tf.square((self.loc - other.loc) / other.scale) +
                            other.log_scale - self.log_scale
                          ),
                          axis=-1) - 0.5 * self.dim


  def _mean(self):
    return self.loc


  def _mode(self):
    return self.loc


  def _stddev(self):
    return self.scale


  @property
  def dimension(self):
    return self.dim
