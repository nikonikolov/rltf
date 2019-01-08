from typing import Callable

import tensorflow as tf

from rltf.tf_utils import tf_utils, tf_cg


class NaturalGradientOptimizer:
  """Optimzier which computes the Truncated Natural Policy Gradient (TNPG) step"""


  def __init__(self, learning_rate, max_kl, cg_iters=10, cg_damping=1e-3, name="NaturalGradient"):
    """
    Args:
      learning_rate: scalar tf.Tensor or float. The step size for the gradient step
      max_kl: float. Maximum allowed KL divergence between the old and the new policy
      cg_iters: int. Number of Conjugate Gradient iterations
      cg_damping: float. Conjugate Gradient damping coefficient
    """
    self.step_size  = learning_rate
    self.max_kl     = max_kl
    self.cg_iters   = cg_iters
    self.cg_damping = cg_damping
    self.name       = name


  def compute_steps(self, pi_gain: tf.Tensor, kl: tf.Tensor, var_list: list) -> list:
    """Compute the Truncated Natural Policy Gradient step
    Args:
      pi_gain: scalar Tensor for the Policy Gradient optimization objective
      kl: scalar Tensor for the KL divergence
      var_list: list of variables to optimize
    Returns:
      list of tf.Tensor. Contains the corresponding update direction for var_list
    """
    with tf.name_scope(self.name):
      steps = self._compute_TNPG_step(pi_gain, kl, var_list)
    return steps


  def apply_steps(self, steps_and_vars, name=None):
    """Apply steps to variables
    Args:
      steps_and_vars: list of tuples of (tf.Tensor, tf.Variable). The first entry contains
        the unscaled step, the second contains the variable to apply the step to
    Returns:
      tf.Operation which applies the steps
    """
    # Compute the updated values for the policy variables
    updates = [v + self.step_size * step for step, v in steps_and_vars]
    pi_vars = [v for _, v in steps_and_vars]

    return tf_utils.assign_vars(pi_vars, updates, name=name)


  def _compute_TNPG_step(self, pi_gain, mean_kl, pi_vars: list) -> list:
    # Compute the policy gradient
    pi_grad = tf.gradients(pi_gain, pi_vars)
    pi_grad = tf_cg.flatten_tensors(pi_grad)     # shape: [None]

    # TODO: Assert pi_grad is not too close to 0
    # assert_op = tf_ops.assert_not_near(pi_grad, 0, message="Policy Gradient near zero")
    # with tf.control_dependencies([assert_op]):
    #   pi_grad = tf.identity(pi_grad)

    # Get the function to compute the Hessian-vector product for the KL Hessian
    f_Hv    = self._fisher_vector_product(mean_kl, pi_vars)

    # Use the Conjugate Gradient to compute H^-1 g
    H_inv_g = tf_cg.conjugate_gradient(f_Av=f_Hv, b=pi_grad, iterations=self.cg_iters,
                                       damping=self.cg_damping)

    # Compute the Natural Policy Gradient step vector and split for variables
    step    = self._compute_step_vector(pi_grad, H_inv_g)
    shapes  = [v.shape.as_list() for v in pi_vars]
    steps   = tf_cg.split_vector(step, shapes)

    return steps


  def _fisher_vector_product(self, mean_kl: tf.Tensor, var_list: list) -> Callable:
    """Get a function that computes the product of the KL Hessian and some vector v.
    Use the fact that Hv = d^2 L / dt^2 v = d/dt (dL/dt) v = d/dt gv
    Args:
      mean_kl: tf.Tensor. The KL divergence between the old and the new policy
      var_list: list of tf.Variables for which to compute gradients
    Returns:
      lambda, which takes as input a vector v and computes the product Hv
    """

    # Compute the gradients of the KL divergence w.r.t. var_list and flatten them
    grads = tf.gradients(mean_kl, var_list)
    grad  = tf_cg.flatten_tensors(grads)     # shape: [None]

    def compute_hvp(v):
      # Compute the dot product between grad and v
      v     = tf.stop_gradient(v)
      gvp   = tf.reduce_sum(grad * v)
      # Compute the matrix-vector product `Hv`, between the Hessian and v and flatten it
      hvps  = tf.gradients(gvp, var_list)
      hvp   = tf_cg.flatten_tensors(hvps)
      hvp   = tf.check_numerics(hvp, message="Invalid Fisher-vector product")
      return hvp

    return compute_hvp


  def _compute_step_vector(self, g: tf.Tensor, H_inv_g: tf.Tensor) -> tf.Tensor:
    """Compute the step vector given by the Natural Policy Gradient
    Args:
      g: tf.Tensor, shape [None]. Contains the policy gradient g
      H_inv_g: tf.Tensor, shape [None]. Contains the product H^-1 g
    Returns:
      tf.Tensor, shape [None]. Contains the full NPG step
    """
    # Compute the dot product `g^T H^-1 g`
    g_H_inv_g = tf.reduce_sum(g * H_inv_g)

    # Compute the corresponding Lagrange multiplier
    lm = tf.sqrt(g_H_inv_g / (2 * self.max_kl))

    # Compute the final step
    step = H_inv_g / lm

    return step
