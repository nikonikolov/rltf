import tensorflow as tf

from rltf.models import BasePG
from rltf.models import tf_utils


class PPO(BasePG):
  """Proximal Policy Optimization Model"""

  def __init__(self, ent_weight, vf_weight, **kwargs):
    super().__init__(**kwargs)

    self.ent_weight = ent_weight
    self.vf_weight  = vf_weight

    # Custom TF placeholders
    self.cliprange_ph = None
    self.old_vf_ph    = None


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

    CLIP_RANGE = self.cliprange_ph

    # Compute the policy gradient loss
    logp      = pi.logp(self.act_ph)
    weights   = tf.exp(logp - self.old_logp_ph)
    pg_loss_1 = weights * self.adv_ph
    pg_loss_2 = tf.clip_by_value(weights, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * self.adv_ph
    pg_loss   = -tf.reduce_mean(tf.minimum(pg_loss_1, pg_loss_2))

    # Compute the policy entropy for Max-Ent learning
    entropy  = tf.reduce_mean(pi.entropy())

    # Compute the Value Function loss
    vf_clip   = tf.clip_by_value(vf, self.old_vf_ph - CLIP_RANGE, self.old_vf_ph + CLIP_RANGE)
    vf_loss_1 = tf.square(vf      - self.ret_ph)
    vf_loss_2 = tf.square(vf_clip - self.ret_ph)
    vf_loss   = 0.5 * tf.reduce_mean(tf.maximum(vf_loss_1, vf_loss_2))

    loss      = pg_loss - self.ent_weight * entropy + self.vf_weight * vf_loss

    # Remember the ops
    # self.ops_dict["loss"]     = loss
    # self.ops_dict["pg_loss"]  = pg_loss
    # self.ops_dict["vf_loss"]  = vf_loss
    # self.ops_dict["entropy"]  = entropy

    # Add metrics to track the training progress
    # Fraction of examples with clipped PG objective
    frac_clip = tf.reduce_mean(tf.cast(tf.greater(tf.abs(weights - 1.0), CLIP_RANGE), dtype=tf.float32))
    # Easy-to-compute approximate estimate of KL between old and new policy
    approxkl  = tf.reduce_mean(self.old_logp_ph - logp)

    # Add summaries
    tf.summary.scalar(tb_name,              loss)
    tf.summary.scalar("train/pg_loss",      pg_loss)
    tf.summary.scalar("train/vf_loss",      vf_loss)
    tf.summary.scalar("train/pi_entropy",   entropy)
    tf.summary.scalar("train/approx_kl",    approxkl)
    tf.summary.scalar("train/frac_clip",    frac_clip)
    tf.summary.scalar("train/vf",           tf.reduce_mean(vf))

    # Add summaries for stdout
    tf.summary.scalar("stdout/pg_loss",     pg_loss)
    tf.summary.scalar("stdout/pi_entropy",  entropy)
    tf.summary.scalar("stdout/approx_kl",   approxkl)

    return loss


  def _build_ph(self):
    super()._build_ph()
    self.cliprange_ph = tf.placeholder(tf.float32, (),     name="cliprange_ph")
    self.old_vf_ph    = tf.placeholder(tf.float32, [None], name="old_vf_ph")
