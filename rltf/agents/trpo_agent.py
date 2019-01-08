import logging
import numpy as np

from rltf.agents import AgentPG


logger = logging.getLogger(__name__)


class AgentTRPO(AgentPG):

  def __init__(self, vf_batch_size, line_search_steps, max_kl, **agent_kwargs):
    """
    Args:
      vf_batch_size: int. Batch size for training the value function
      line_search_steps: int. Number of max line search iterations
      max_kl: float. Maximum allowed KL divergence between the old and the new policy
    """
    super().__init__(**agent_kwargs, max_kl=max_kl)

    self.max_kl             = max_kl
    self.vf_batch_size      = vf_batch_size
    self.line_search_steps  = line_search_steps


  def _run_train_step(self, t):
    # Get all collected data
    batch     = self.buffer.get_data()
    feed_dict = self._get_feed_dict(batch, t)

    # Update the old policy to match the current one
    self.sess.run(self.model.update_old_pi)

    # Compute the policy surrogate gain before the update and the TNPG step
    pi_gain_lo, _ = self.sess.run([self.model.pi_gain, self.model.step_op], feed_dict=feed_dict)

    # Perform line search for the new policy
    self._line_search(pi_gain_lo, feed_dict)

    # Train the value function
    for _ in range(self.vf_iters):
      # Iterate over all data in the buffer in mini-batches
      for batch in self.buffer.iterate(batch_size=self.vf_batch_size, shuffle=True):
        if self._terminate:
          return

        feed_dict = self._get_feed_dict(batch, t)

        # Run a value function training step
        self.sess.run(self.model.train_vf, feed_dict=feed_dict)

    # Run the summary op to log the changes from the update if necessary
    self._run_summary_op(t, feed_dict)


  def _line_search(self, pi_gain_lo, feed_dict):
    """Perform line search for the new policy.
    Args:
      pi_gain_lo: float. Policy surrogate gain before the update
      feed_dict: dict with data to feed to the session in order to compute the line search metrics
    """
    step_size = 1.0

    for _ in range(self.line_search_steps):
      if self._terminate:
        return

      # Update the policy
      self.sess.run(self.model.update_pi, feed_dict={self.model.step_size_ph: step_size})

      # Compute the policy surrogate gain and the KL divergence
      kl, pi_gain = self.sess.run([self.model.mean_kl, self.model.pi_gain], feed_dict=feed_dict)

      if not np.isfinite(kl) or not np.isfinite(pi_gain):
        logger.warning("Non-finite loss values: kl=%f, pi_gain=%f. Shrinking step", kl, pi_gain)
      elif kl > self.max_kl * 1.5:
        logger.debug("Violated KL constraint. Shrinking step")
      elif pi_gain < pi_gain_lo:
        logger.debug("Surrogate objective did not improve. Shrinking step")
      else:
        logger.debug("Stepsize OK")
        break

      # Shrink step size
      step_size = step_size * 0.5

    else:
      logger.info("Line search could not compute a good step")
      # Reset pi to its initial state
      self.sess.run(self.model.reset_pi)
