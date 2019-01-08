from rltf.agents import AgentPG


class AgentPPO(AgentPG):

  def __init__(self, train_steps, batch_size, clip_range, **agent_kwargs):
    """
    Args:
      train_steps: int. Number of training steps per epoch
      batch_size: int. Batch size for training the model
      clip_range: rltf.schedules.Schedule. Clipping range value for PPO and VF objective
    """
    super().__init__(**agent_kwargs)

    self.train_steps  = train_steps
    self.batch_size   = batch_size
    self.clip_range   = clip_range


  def _get_feed_dict(self, batch, t):
    feed_dict = super()._get_feed_dict(batch, t)

    # Append the VF and cliprange
    feed_dict[self.model.old_vf_ph]     = batch["vf"]
    feed_dict[self.model.cliprange_ph]  = self.clip_range.value(t)

    return feed_dict


  def _run_train_step(self, t):

    for _ in range(self.train_steps):

      # Iterate over all data in the buffer in mini-batches
      for batch in self.buffer.iterate(batch_size=self.batch_size, shuffle=True):
        if self._terminate:
          return

        feed_dict = self._get_feed_dict(batch, t)

        # Run a policy gradient step and a value function training step
        self.sess.run(self.model.train_op, feed_dict=feed_dict)

    # Run the summary op to log the changes from the update if necessary
    self._run_summary_op(t, feed_dict)
