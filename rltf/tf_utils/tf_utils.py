import logging
import tensorflow as tf


logger = logging.getLogger(__name__)


# ------------------------------------ OPERATIONS ------------------------------------


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


def scope_vars(var_list, scope):
  """
  Args:
    var_list: list of `tf.Variable`s. Contains all variables that should be searched
    scope: str. Scope of the variables that should be selected
  """
  return [v for v in var_list if scope in v.name]


def normalize(x, training, momentum=0.0):
  """Normalize a tensor along the batch dimension. Normalization is done using the statistics of the
  current batch (in training mode) or based on running mean and variance (in inference mode).
  Args:
    x: tf.Tensor, shape.ndims == 2. Input tensor
    training: tf.Tensor or bool. Whether to return the output in training mode (normalized with
      statistics of the current batch) or in inference mode (normalized with moving statistics)
    momentum: float. Momentum for the moving average.
  """
  assert x.shape.ndims == 2

  kwargs = dict(axis=-1, center=False, scale=False, trainable=True, training=training, momentum=momentum)

  ops = tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
  i   = len(ops)

  x = tf.layers.batch_normalization(x, **kwargs)

  # Get the batch norm update ops and remove them from the global list
  update_ops = ops[i:]
  del ops[i:]

  # Update the moving mean and variance before returning the output
  with tf.control_dependencies(update_ops):
    x = tf.identity(x)
  return x


def preprocess_input(x, norm=True, training=None, momentum=0.0):
  """Preprocess input observations by optionally normalizing them.
  Args:
    x: tf.Tensor. Input tensor. When image observations, `shape.ndims` must be `4` and dtype must be
      `uint8`. When low-dimensional observations, `shape.ndims` must be `2` and dtype must be float
    norm: bool. If True, normalize the tensor
    training: tf.Tensor or bool. Required only for low-dimensional tensors. See normalize()
    momentum: float. See normalize()
  """
  # Image input
  if x.shape.ndims == 4 and x.dtype.base_dtype == tf.uint8:
    x = tf.cast(x, tf.float32)
    if norm:
      x = x / 255.0
  # Low-dimensional 2D input
  elif x.shape.ndims == 2 and x.dtype.base_dtype == tf.float32 or x.dtype.base_dtype == tf.float64:
    if norm:
      assert training is not None
      # Normalize observations
      x = normalize(x, training, momentum)
  else:
    raise ValueError("Invalid observation shape and type")
  return x


# ------------------------------------ INITIALIZERS ------------------------------------


def init_he_relu():
  """
  Returns:
    A normal distribution initializer with std = sqrt(2.0 / fan_in), where fan_in
    is the size of the variable
  """
  # variance_scaling_initializer: https://www.tensorflow.org/api_docs/python/tf/variance_scaling_initializer
  return tf.variance_scaling_initializer(scale=2.0, mode="fan_in", distribution="normal")


def init_glorot_normal():
  return tf.glorot_normal_initializer()


def init_default():
  return None


def init_dqn():
  """Return the initializer used in DQN and its improvements"""
  return tf.variance_scaling_initializer(scale=1./3.0, mode="fan_in", distribution="uniform")
