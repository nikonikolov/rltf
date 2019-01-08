import tensorflow as tf

from rltf.models    import BasePG
from rltf.tf_utils  import tf_utils


class TRPO(BasePG):
  """Trust Region Policy Optimization Model"""

  def __init__(self, ent_weight, **kwargs):
    """
    Args:
      ent_weight: float. Coefficient for Entropy Maximization in the surrogate objective
    """
    super().__init__(**kwargs)

    self.ent_weight = ent_weight

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

    # Build the Value Function train op
    train_vf    = self._build_vf_train_op(losses["vf_loss"], vf_vars, name="train_vf")

    # Compute the Truncated Natural Policy Gradient step
    train_pi    = self._build_pi_train_op(losses["pi_gain"], losses["mean_kl"], pi_vars, name="save_steps")

    # Assign operators
    reset_pi      = tf_utils.assign_vars(pi_vars, old_pi_vars,  name="reset_pi")
    update_old_pi = tf_utils.assign_vars(old_pi_vars, pi_vars,  name="update_old_pi")

    self.step_op        = train_pi["step_op"]     # Op which computes the TNPG step
    self.update_pi      = train_pi["update_pi"]   # Op which updates pi with the TNPG step
    self.reset_pi       = reset_pi                # Op which resets pi to old_pi
    self.update_old_pi  = update_old_pi           # Op which updates old_pi to pi
    self.mean_kl        = losses["mean_kl"]
    self.pi_gain        = losses["pi_gain"]
    self.train_vf       = train_vf

    # Compute the train and eval actions
    self.train_dict = self._act_train(pi, vf, name="a_train")
    self.eval_dict  = self._act_eval(pi, name="a_eval")

    self._vars   = pi_vars + vf_vars


  def _build_pi_train_op(self, pi_gain, mean_kl, pi_vars, name):
    # Build the Natural Gradient optimizer
    pi_opt = self.pi_opt_conf.build()

    # Compute the TNPG step
    steps = pi_opt.compute_steps(pi_gain=pi_gain, kl=mean_kl, var_list=pi_vars)

    # Get variables to save the computed steps into
    step_vars = self._get_step_variables(pi_vars, scope="policy_steps")
    # Get an Op which saves the steps
    save_steps  = tf_utils.assign_vars(step_vars, steps, name=name)

    # Get an Op which applies the computed steps
    steps_and_vars  = list(zip(step_vars, pi_vars))
    apply_steps     = pi_opt.apply_steps(steps_and_vars)

    return dict(step_op=save_steps, update_pi=apply_steps)


  def _build_vf_train_op(self, loss, vf_vars, name=None):
    vf_opt    = self.vf_opt_conf.build()
    train_vf  = vf_opt.minimize(loss, var_list=vf_vars, name=name)
    return train_vf


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


  def _get_step_variables(self, pi_vars, scope):
    with tf.variable_scope(scope):
      step_vars = [tf.get_variable(name=pi_var.name[6:-2], shape=pi_var.shape, dtype=pi_var.dtype,
                                   initializer=tf.zeros_initializer()) for pi_var in pi_vars]
    return step_vars
