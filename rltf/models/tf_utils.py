import logging
import tensorflow as tf

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


def init_default():
  return None
