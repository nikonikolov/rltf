import tensorflow as tf

from rltf.models import BasePG
from rltf.models import tf_utils


class REINFORCE(BasePG):
  """Vanilla Policy Gradient Model"""

  def build(self):

    # Build the input placeholders
    self._build_ph()

    # Preprocess the observation
    obs_t   = tf_utils.preprocess_input(self.obs_ph, norm=self.obs_norm, training=self.training)

    # Construct the policy and the value function networks
    pi      = self._pi_model(obs_t, scope="policy")
    vf      = self._vf_model(obs_t, scope="value_fn")

    # Compute the loss
    loss    = self._compute_loss(pi, vf, tb_name="train/loss")

    pi_vars = self._trainable_variables(scope="policy")
    vf_vars = self._trainable_variables(scope="value_fn")

    # Build the optimizer and the train op
    train_op  = self._build_train_op(loss, pi_vars, vf_vars, name="train_op")

    # Compute the train and eval actions
    self.train_dict = self._act_train(pi, vf, name="a_train")
    self.eval_dict  = self._act_eval(pi, name="a_eval")

    self._vars        = pi_vars + vf_vars
    self.train_op     = train_op


  def _compute_loss(self, pi, vf, tb_name):
    logp    = pi.logp(self.act_ph)
    pg_loss = - tf.reduce_mean(logp * self.adv_norm)
    vf_loss = tf.losses.mean_squared_error(self.ret_ph, vf)
    loss    = vf_loss + pg_loss

    # Remember the ops
    self.ops_dict["loss"]    = loss
    self.ops_dict["pg_loss"] = pg_loss
    self.ops_dict["vf_loss"] = vf_loss

    # Easy-to-compute approximate estimates of KL and entropy (assume uniform weights)
    approxkl  = tf.reduce_mean(self.old_logp_ph - logp)
    approxent = tf.reduce_mean(-logp)

    # Add TensorBoard summaries
    tf.summary.scalar(tb_name,               loss)
    tf.summary.scalar("train/pg_loss",    pg_loss)
    tf.summary.scalar("train/vf_loss",    vf_loss)
    tf.summary.scalar("train/approx_ent", approxent)
    tf.summary.scalar("train/approx_kl",  approxkl)

    # Add summaries for stdout
    tf.summary.scalar("stdout/approx_ent", approxent)
    tf.summary.scalar("stdout/approx_kl",  approxkl)

    return loss
