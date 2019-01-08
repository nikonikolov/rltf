from typing import Callable

import tensorflow as tf

from rltf.models    import BasePG
from rltf.tf_utils  import tf_utils, tf_cg


class TRPO(BasePG):
  """Trust Region Policy Optimization Model"""

  def __init__(self, max_kl, cg_iters, cg_damping, ent_weight, **kwargs):
    """
    Args:
      max_kl: float. Maximum allowed KL divergence between the old and the new policy
      cg_iters: int. Number of Conjugate Gradient iterations
      cg_damping: float. Conjugate Gradient damping coefficient
      ent_weight: float. Coefficient for Entropy Maximization in the surrogate objective
    """
    super().__init__(**kwargs)

    self.ent_weight = ent_weight
    self.max_kl     = max_kl
    self.cg_iters   = cg_iters
    self.cg_damping = cg_damping

    # Custom TF placeholders
    self.step_size_ph = None

    # Custom TF Ops
    self.step_op        = None
    self.update_pi      = None
    self.reset_pi       = None
    self.update_old_pi  = None
    self.mean_kl        = None
    self.pi_gain        = None
    self.train_vf       = None


  def build(self):

    # Build the input placeholders
    self._build_ph()

    # Preprocess the observation
    obs_t   = tf_utils.preprocess_input(self.obs_ph, norm=self.obs_norm, training=self.training)

    # Construct the policy and the value function networks
    pi      = self._pi_model(obs_t, scope="policy")
    old_pi  = self._pi_model(obs_t, scope="old_policy")
    vf      = self._vf_model(obs_t, scope="value_fn")

    # Compute all losses/objectives
    losses  = self._compute_losses(pi, old_pi, vf)

    pi_vars     = self._trainable_variables(scope="policy")
    old_pi_vars = self._trainable_variables(scope="old_policy")
    vf_vars     = self._trainable_variables(scope="value_fn")
    step_vars   = self._get_step_variables(pi_vars, scope="policy_steps")

    # Build the Value Function train op
    train_vf    = self._build_vf_train_op(losses["vf_loss"], vf_vars, name="train_vf")

    # Compute the Truncated Natural Policy Gradient step
    steps       = self._compute_TNPG_step(losses, pi_vars)

    # Compute the updated values for the policy variables
    new_pi_vars = [pi_var + self.step_size_ph * step for pi_var, step in zip(pi_vars, step_vars)]

    # Assign operators
    save_steps    = tf_utils.assign_vars(step_vars, steps,      name="save_steps")
    update_pi     = tf_utils.assign_vars(pi_vars, new_pi_vars,  name="update_pi")
    reset_pi      = tf_utils.assign_vars(pi_vars, old_pi_vars,  name="reset_pi")
    update_old_pi = tf_utils.assign_vars(old_pi_vars, pi_vars,  name="update_old_pi")

    self.step_op        = save_steps          # Op which computes the TNPG step
    self.update_pi      = update_pi           # Op which updates pi with the TNPG step
    self.reset_pi       = reset_pi            # Op which resets pi to old_pi
    self.update_old_pi  = update_old_pi       # Op which updates old_pi to pi
    self.mean_kl        = losses["mean_kl"]
    self.pi_gain        = losses["pi_gain"]
    self.train_vf       = train_vf

    # Compute the train and eval actions
    self.train_dict = self._act_train(pi, vf, name="a_train")
    self.eval_dict  = self._act_eval(pi, name="a_eval")

    self._vars   = pi_vars + vf_vars


  def _compute_TNPG_step(self, losses: dict, pi_vars: list) -> list:
    """Compute the Truncated Natural Policy Gradient step
    Args:
      losses: dict. Must contain the PG optimization objective and the KL divergence
      pi_vars: list of policy variables to optimize
    Returns:
      list of tf.Tensor. Contains the corresponding update direction for pi_vars
    """

    mean_kl = losses["mean_kl"]
    pi_gain = losses["pi_gain"]

    # Compute the policy gradient
    pi_grad = tf.gradients(pi_gain, pi_vars)
    pi_grad = tf_cg.flatten_tensors(pi_grad)     # shape: [None]

    # # Assert pi_grad is not too close to 0
    # assert_op = tf_ops.assert_not_near(pi_grad, 0, message="Policy Gradient near zero")
    # with tf.control_dependencies([assert_op]):
    #   pi_grad = tf.identity(pi_grad)

    # Get the function to compute the Hessian-vector product for the KL Hessian
    f_Hv    = self._hessian_vector_product(mean_kl, pi_vars)

    # Use the Conjugate Gradient to compute H^-1 g
    H_inv_g = tf_cg.conjugate_gradient(f_Av=f_Hv, b=pi_grad, iterations=self.cg_iters)

    # Compute the Natural Policy Gradient step vector and split for variables
    step    = self._compute_step_vector(pi_grad, H_inv_g)
    shapes  = [v.shape.as_list() for v in pi_vars]
    steps   = tf_cg.split_vector(step, shapes)

    return steps


  def _hessian_vector_product(self, mean_kl: tf.Tensor, var_list: list) -> Callable:
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
      gvp   = tf.reduce_sum(grad * v)

      # Compute the matrix-vector product `Hv`, between the Hessian and v and flatten it
      hvps  = tf.gradients(gvp, var_list)
      hvp   = tf_cg.flatten_tensors(hvps)
      # Apply damping
      hvp   = hvp + self.cg_damping * v
      hvp   = tf.check_numerics(hvp, message="Invalid Hessian-vector product")
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


  def _compute_losses(self, pi, old_pi, vf):
    # Compute the KL divergence between the two policies
    mean_kl = tf.reduce_mean(old_pi.kl(pi))

    # Compute the policy gradient maximization objective: advantage * p_new / p_old
    pg_objective = self.adv_norm * tf.exp(pi.logp(self.act_ph) - self.old_logp_ph)
    # pg_objective = self.adv_norm * tf.exp(pi.logp(self.act_ph) - old_pi.logp(self.act_ph))
    pg_objective = tf.reduce_mean(pg_objective)

    # Compute the policy entropy for Max-Ent learning
    entropy   = tf.reduce_mean(pi.entropy())

    # Compute the final optimization objective
    objective = pg_objective + self.ent_weight * entropy

    # Compute the Value Function loss
    vf_loss   = tf.losses.mean_squared_error(self.ret_ph, vf)

    # Remember the ops
    # self.ops["surr_gain"] = pg_objective

    # Add TB summaries
    tf.summary.scalar("train/surr_gain",    objective)
    tf.summary.scalar("train/pi_gain",      pg_objective)
    # tf.summary.scalar("train/vf_loss",      vf_loss)
    tf.summary.scalar("train/pi_entropy",   entropy)
    tf.summary.scalar("train/kl",           mean_kl)

    # Add summaries for stdout
    tf.summary.scalar("stdout/surr_gain",    objective)
    tf.summary.scalar("stdout/pi_gain",      pg_objective)
    tf.summary.scalar("stdout/pi_entropy",   entropy)
    tf.summary.scalar("stdout/kl",           mean_kl)

    return dict(pi_gain=objective, mean_kl=mean_kl, vf_loss=vf_loss)


  def _build_ph(self):
    super()._build_ph()
    self.step_size_ph = tf.placeholder(tf.float32, (), name="step_size_ph")


  def _get_step_variables(self, pi_vars, scope):
    with tf.variable_scope(scope):
      step_vars = [tf.get_variable(name=pi_var.name[6:-2], shape=pi_var.shape, dtype=pi_var.dtype,
                                   initializer=tf.zeros_initializer()) for pi_var in pi_vars]
    return step_vars


  def _build_vf_train_op(self, loss, vf_vars, name=None):
    vf_opt    = self.vf_opt_conf.build()
    train_vf  = vf_opt.minimize(loss, var_list=vf_vars, name=name)
    return train_vf
