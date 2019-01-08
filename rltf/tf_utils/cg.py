from typing import Callable

import numpy as np
import tensorflow as tf


def conjugate_gradient(f_Av: Callable, b: tf.Tensor, iterations: int, damping=0.0,
                       tolerance=1e-10) -> tf.Tensor:
  """Compute the solution to Ax=b using the Conjugate Gradient method. Uses tf operations
  Args:
    b: tf.Tensor, shape `[None]`
    f_Av: lambda. Takes a vector `v` as argument and computes the matrix-vector product `Av`
    iterations: int. Number of iterations
    damping: float. CG damping coefficient
    tolerance: float or tf.Tensor. Return `x`, if the square of the residual gets below this tolerance
  Returns:
    tf.Tensor of the same shape as `b` which contains the solution
  """
  assert b.shape.ndims == 1
  assert tolerance > 0.0

  def cg_body(i, x, r, p, rTr):
    Ap      = f_Av(p)
    if damping > 0.0:
      Ap      = Ap + damping * p
    alpha   = rTr / tf.reduce_sum(p * Ap)
    x       = x + alpha * p
    r       = r + alpha * Ap
    rTr_    = tf.reduce_sum(tf.square(r))
    # Make sure division by 0 is avoided
    beta    = rTr_ / tf.maximum(rTr, tolerance)
    p       = - r + beta * p
    i       = tf.add(i, 1)

    return [i, x, r, p, rTr_]

  #pylint: disable=unused-argument
  def cg_cond(i, x, r, p, rTr):
    return tf.greater(rTr, tolerance)

  # Initial CG values
  x   = tf.zeros_like(b, dtype=tf.float32)  # [None, 1]
  r   = -b
  p   = -r
  i   = tf.constant(0)
  rTr = tf.reduce_sum(tf.square(r))

  # _, x, _, _, _ = tf.while_loop(cg_cond, cg_body, loop_vars=[i, x, r, p, rTr])

  loop_vars = [i, x, r, p, rTr]

  # Use unrolled loop; TF does not allow computing gradients inside a tf.while_loop
  for _ in range(iterations):
    cond      = cg_cond(*loop_vars)
    update    = lambda: cg_body(*loop_vars)
    identity  = lambda: loop_vars
    loop_vars = tf.cond(pred=cond, true_fn=update, false_fn=identity)

  x = loop_vars[1]

  return tf.check_numerics(x, message="Invalid Conjugate Gradient solution", name="cg_x")


def conjugate_gradient_np(f_Av, b, iterations, tolerance=1e-10):
  """Compute the solution to Ax=b using the Conjugate Gradient method. Uses numpy operations
  Args:
    b: np.array, shape `[None]`
    f_Av: lambda. Takes a vector `v` as argument and computes the matrix-vector product `Av`
    iterations: int. Number of iterations
    tolerance: float. Return `x`, if the square of the residual gets below this tolerance
  Returns:
    np.array of the same shape as `b` which contains the solution
  """
  dtype = b.dtype

  x = np.zeros_like(b, dtype=dtype)  # [None, 1]
  r = np.array(-b, dtype=dtype)
  p = np.array(-r, dtype=dtype)      #pylint: disable=invalid-unary-operand-type

  for _ in range(iterations):
    Ap      = np.asarray(f_Av(p), dtype=dtype)
    rTr     = np.dot(r.T, r)
    alpha   = rTr / np.dot(p.T, Ap)
    x       = x + alpha * p
    r       = r + alpha * Ap
    rTr_    = np.dot(r.T, r)
    beta    = rTr_ / rTr
    p       = - r + beta * p
    rTr     = rTr_

    if rTr < tolerance:
      break

  return x


def flatten_tensors(tensor_list: list) -> tf.Tensor:
  """Flatten the tensors in tensor_list and concatenate them in a vector
  Args:
    tensors_list: list of tf.Tensors.
  Returns:
    tf.Tensor of shape `[None]`, which contains the flattened tensors in tensor_list in order
  """
  for tensor in tensor_list:
    assert isinstance(tensor, tf.Tensor)
  flat = tf.concat([tf.reshape(tensor, [-1]) for tensor in tensor_list], axis=0)
  return flat


def split_vector(v: tf.Tensor, shapes: list) -> list:
  """Slice and reshape a vector such that the shapes of the slices match the ones in shapes_list
  or of the tensors in tensor_list. Only one of `tensor_list` and `shapes` must be specified
  Args:
    v: tf.Tensor. The vector to be split
    shapes: list of shapes to split into
  """
  # assert (tensor_list is None) != (shapes is None)
  assert v.shape.ndims == 1

  # shapes  = [tensor.shape.as_list() for tensor in tensor_list] if shapes is None else shapes
  # shapes  = [tensor.shape.as_list() for tensor in tensor_list]
  sizes   = [np.prod(shape) for shape in shapes]

  assert v.shape.ndims == 1
  assert np.sum(sizes) == v.shape.as_list()[0]

  flats   = tf.split(v, sizes)
  tensors = [tf.reshape(tensor, shape) for tensor, shape in zip(flats, shapes)]

  return tensors
