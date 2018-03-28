import logging
import tensorflow as tf

import rltf.conf

logger = logging.getLogger(__name__)


def assign_vars(dest_vars, source_vars, weight=1.0, name=None):
  """Create a `tf.Op` that assigns the values of source_vars to dest_vars.
  `source_vars` and `dest_vars` must have variables with matching names,
  but do not need to be sorted.
  The assignment operation is:
  ```
  dest_var = weight*source_var + (1-weight)*dest_var
  ```

  Args:
    dest_vars: list of tf.Variables. Holds the variables that will be updated
    source_vars: list of tf.Variables. Holds the source values
    weight: int. Weight to use in the above expression
    name: string. Optional name for the returned operation
  Returns:
    `tf.Op` that performs the assignment
  """
  assert weight <= 1.0
  assert weight >  0.0
  assert len(source_vars) == len(dest_vars)

  update_ops = []

  logger.debug("Assigning tf values as:")
  for s_var, d_var in zip(source_vars, dest_vars):
    logger.debug(d_var.name + " := " + s_var.name)
    if weight == 1.0:
      update_ops.append(tf.assign(d_var, s_var))
    else:
      update_ops.append(tf.assign(d_var, (1.-weight)*d_var + weight*s_var))

  assert len(update_ops) > 0

  return tf.group(*update_ops, name=name)



def huber_loss(x, delta=1.0):
  """Apply the function:
  ```
  x^2 if |x| < delta else delta*(|x| - 0.5*delta)
  ```
  """
  abs_x = tf.abs(x)
  return tf.where(
    abs_x < delta,
    tf.square(x) * 0.5,
    delta * (abs_x - 0.5 * delta)
  )


def init_he_relu():
  """
  Returns:
    A normal distribution initializer with std = sqrt(2.0 / fan_in), where fan_in
    is the size of the variable
  """
  # variance_scaling_initializer: https://www.tensorflow.org/api_docs/python/tf/variance_scaling_initializer
  return tf.variance_scaling_initializer(scale=2.0, mode="fan_in", distribution="normal")


def init_glorot_normal():
  return tf.glorot_normal_initializer(seed=rltf.conf.SEED)


def init_default():
  return None


def cholesky_inverse(A):
  """Compute the inverse of `A` using Choselky decomposition. NOTE: `A` must be
  symmetric positive definite. This method of inversion is not completely stable since
  tf.cholesky is not always stable. Might raise `tf.errors.InvalidArgumentError`
  """
  N     = tf.shape(A)[0]
  L     = tf.cholesky(A)
  L_inv = tf.matrix_triangular_solve(L, tf.eye(N))
  A_inv = tf.matmul(L_inv, L_inv, transpose_a=True)
  return A_inv


def sherman_morrison_inverse(A_inv, u, v):
  """Compute the inverse of (A + uv^T) using Sherman-Morrison formula:
  https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
  Args:
    A_inv: tf.Tensor. The inverse of A or batch. Last two dimensions should have shape [N, N]
    u: tf.Tensor. (Batch of) column vector(s). Last two dimensions should have shape [N, 1]
    v: tf.Tensor. (Batch of) column vector(s). Last two dimensions should have shape [N, 1]
  Returns: (A + uv^T)^{-1} with the same shape as `A_inv`
  """
  assert u.shape.as_list()[-1] == 1 and len(u.shape) >= 2
  assert v.shape.as_list()[-1] == 1 and len(v.shape) >= 2

  A_inv_u = tf.matmul(A_inv, u)
  num     = tf.matmul(A_inv_u, tf.matmul(v, A_inv, transpose_a=True))
  denom   = tf.matmul(v, A_inv_u, transpose_a=True)
  denom   = 1 + tf.squeeze(denom, axis=[-2, -1])
  inverse = A_inv - num / denom

  return inverse


def woodburry_inverse(A_inv, U, V):
  """Compute the inverse of (A + UV) using Woodburry formula:
  `(A + UV)^-1 = A^-1 - A^-1 U (I + V A^-1 U)^-1 V A^-1`. For details see:
  https://en.wikipedia.org/wiki/Woodbury_matrix_identity
  Args:
    A_inv: tf.Tensor. The inverse of A, `shape=[N, N]`
    U: tf.Tensor, `shape=[N, M]`
    V: tf.Tensor. `shape=[M, N]`
  Returns: (A + UV)^-1 with the same shape and dtype as `A_inv`
  """

  # NOTE: Must make sure to use double precision. Otherwise results are very inaccurate
  A_inv_64  = tf.cast(A_inv, tf.float64) if  A_inv.dtype.base_dtype == tf.float32 else A_inv
  U_64      = tf.cast(U,     tf.float64) if      U.dtype.base_dtype == tf.float32 else U
  V_64      = tf.cast(V,     tf.float64) if      V.dtype.base_dtype == tf.float32 else V

  assert  A_inv_64.dtype.base_dtype == tf.float64
  assert      U_64.dtype.base_dtype == tf.float64
  assert      V_64.dtype.base_dtype == tf.float64

  A_inv_U = tf.matmul(A_inv_64, U_64)
  V_A_inv = tf.matmul(V_64, A_inv_64)
  I       = tf.eye(tf.shape(V)[0], dtype=tf.float64)
  inverse = tf.matrix_inverse(I + tf.matmul(V_64, A_inv_U))
  inverse = tf.matmul(A_inv_U, inverse)
  inverse = tf.matmul(inverse, V_A_inv)
  inverse = A_inv_64 - inverse

  assert  inverse.dtype.base_dtype == tf.float64
  inverse = tf.cast(inverse, tf.float32) if A_inv.dtype.base_dtype == tf.float32 else inverse
  assert  inverse.dtype.base_dtype == A_inv.dtype.base_dtype

  return inverse
