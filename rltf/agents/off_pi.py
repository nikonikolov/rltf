import logging
import threading
import tensorflow as tf

from rltf.agents.agent import Agent


logger = logging.getLogger(__name__)


class OffPolicyAgent(Agent):
  """The base class for Off-policy agents. train() and eval() procedures *cannot* run in parallel
  (because only 1 environment used). Assumes usage of target network and replay buffer.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.replay_buf = None
    self.update_target_freq = None
    self._terminate = False
    self._save_buf  = False
    self.threads = None


  def train(self):
    # Start threads
    for t in self.threads:
      t.start()

    # Wait for threads
    stop = False
    while not stop:
      try:
        for t in self.threads:
          t.join()
        stop = True
      except KeyboardInterrupt:
        # Confirm the kill
        if not self._kill_confirmed():
          logger.info("CONTINUING EXECUTION")
          continue
        logger.info("EXITING")
        self._terminate = True
        for t in self.threads:
          t.join()
        stop = True


  def eval(self):
    # Use a thread to avoid accidental KeyboardInterrupt
    eval_thread = threading.Thread(name='eval_thread', target=self._eval)

    eval_thread.start()

    # Wait for eval thread
    stop = False
    while not stop:
      try:
        eval_thread.join()
        stop = True
      except KeyboardInterrupt:
        # Confirm the kill
        if not self._kill_confirmed():
          logger.info("CONTINUING EXECUTION")
          continue
        logger.info("EXITING")
        self._terminate = True
        eval_thread.join()
        stop = True


  def _kill_confirmed(self):
    """Check if Ctrl+C was genuine and if the buffer should be saved.
    Returns:
      `bool`. If True, kill. If False, continue
    """
    if self.confirm_kill:
      y = ''
      while True:
        y = input("Do you really want to exit? [y/n]. If you want to exit and save buffer, type 's'")
        if y not in ['y', 'n', 's']:
          print("Response not recognized. Expected 'y', 'n' or 's'.")
        else:
          break
      if y == 'n':
        return False
      elif y == 's':
        self._save_buf = True
    return True


  def _restore(self, graph):
    self.replay_buf.restore(self.model_dir)


  def _save(self):
    if self._save_buf:
      self.replay_buf.save(self.model_dir)


  def _run_train_step(self, t, run_summary):
    # Compose feed_dict
    batch     = self.replay_buf.sample(self.batch_size)
    feed_dict = self._get_feed_dict(batch, t)

    # Wait for synchronization if necessary
    self._wait_act_chosen()

    # Run a training step
    if run_summary:
      self.summary, _ = self.sess.run([self.summary_op, self.model.train_op], feed_dict=feed_dict)
    else:
      self.sess.run(self.model.train_op, feed_dict=feed_dict)

    # Update target network
    if t % self.update_target_freq == 0:
      self.sess.run(self.model.update_target)


  def _eval(self):

    logger.info("Starting evaluation")

    # Set the monitor in evaluation mode
    self.env_monitor.mode = 'e'

    start_step  = self.sess.run(self.t_eval) + 1
    stop_step   = start_step + self.eval_len
    stop_step   = stop_step - stop_step % self.eval_len + 1   # Restore point might be the middle of eval

    obs = self.reset()

    for t in range(start_step, stop_step):
      if self._terminate:
        break

      # Increment the current eval step
      self.sess.run(self.t_eval_inc)

      action = self._action_eval(obs, t)
      next_obs, _, done, _ = self.env.step(action)

      # Reset the environment if end of episode
      if done:
        next_obs = self.reset()
      obs = next_obs

      if t % self.log_freq == 0:
        # Log the statistics
        self.env_monitor.log_stats(t)

        # Add a TB summary
        summary = tf.Summary()
        summary.value.add(tag="eval/mean_ep_rew", simple_value=self.env_monitor.mean_ep_rew)
        self.tb_eval_writer.add_summary(summary, t)

    # Set the monitor back to train mode
    self.env_monitor.mode = 't'

    logger.info("Evaluation finished")


  def _wait_act_chosen(self):
    raise NotImplementedError()


  def _action_train(self, state, t):
    """Return action selected by the agent for a training step
    Args:
      state: np.array. Current state
      t: int. Current timestep
    """
    raise NotImplementedError()


  def _action_eval(self, state, t):
    """Return action selected by the agent for an evaluation step
    Args:
      state: np.array. Current state
      t: int. Current timestep
    """
    raise NotImplementedError()



class ParallelOffPolicyAgent(OffPolicyAgent):
  """Runs the environment and trains the model in parallel using separate threads.
  Provides an easy way to synchronize between the threads. Speeds up training by 20-50%
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Create synchronization events
    self._act_chosen = threading.Event()
    self._train_done = threading.Event()

    self._act_chosen.clear()
    self._train_done.set()

    env_thread    = threading.Thread(name='env_thread', target=self._run_env)
    nn_thread     = threading.Thread(name='net_thread', target=self._train_model)
    self.threads  = [nn_thread, env_thread]


  def _run_env(self):
    """Thread for running the environment. Must call `self._wait_train_done()`
    before selcting an action (by running the model). This ensures that the
    `self._train_model()` thread has finished the training step. After action
    is selected, it must call `self._signal_act_chosen()` to allow
    `self._train_model()` thread to start a new training step
    """

    obs = self.reset()

    for t in range (self.start_step, self.stop_step+1):
      if self._terminate:
        self._signal_act_chosen()
        break

      # Get an action to run
      if self.learn_started:
        try:
          action = self._action_train(obs, t)
        except tf.errors.InvalidArgumentError as e:
          logger.exception("Numerical Exception: %s", e)
          action = self.env.action_space.sample()
          # Terminate program
          self._terminate = True

      # Choose random action if learning has not started
      else:
        action = self.env.action_space.sample()

      # Signal to net_thread that action is chosen
      self._signal_act_chosen()

      # Increment the TF timestep variable
      self.sess.run(self.t_train_inc)

      # Run action
      next_obs, reward, done, _ = self.env.step(action)

      # Store the effect of the action taken upon obs
      self.replay_buf.store(obs, action, reward, done)

      self._log_stats(t)

      # Wait until net_thread is done
      self._wait_train_done()

      # Reset the environment if end of episode
      if done:
        next_obs = self.reset()
      obs = next_obs

      # Stop and run evaluation procedure
      if self.eval_len > 0 and t % self.eval_freq == 0:
        self.eval()
        # Reset the environment on return
        obs = self.reset()


  def _train_model(self):
    """Thread for trianing the model. Must call `self._wait_act_chosen()`
    before trying to run a training step on the model. This ensures that the
    `self._run_env()` thread has finished selcting an action (by running the model).
    After training step is done, it must call `self._signal_train_done()` to allow
    `self._run_env()` thread to select a new action
    """

    for t in range (self.start_step, self.stop_step+1):
      if self._terminate:
        self._signal_train_done()
        break

      if (t >= self.warm_up and t % self.train_freq == 0):

        self.learn_started = True

        # Run a training step
        run_summary = t % self.log_freq + self.train_freq >= self.log_freq
        self._run_train_step(t, run_summary=run_summary)

      else:
        # Synchronize
        self.replay_buf.wait_stored()
        self.replay_buf.signal_sampled()
        self._wait_act_chosen()

      if self.save_freq > 0 and t % self.save_freq == 0:
        self.save()

      self._signal_train_done()


  def _wait_act_chosen(self):
    # Wait until an action is chosen to be run
    while not self._act_chosen.is_set():
      self._act_chosen.wait()
    self._act_chosen.clear()

  def _wait_train_done(self):
    # Wait until training step is done
    while not self._train_done.is_set():
      self._train_done.wait()
    self._train_done.clear()

  def _signal_act_chosen(self):
    # Signal that the action is chosen and the TF graph is safe to be run
    self._act_chosen.set()

  def _signal_train_done(self):
    # Signal to env thread that the training step is done running
    self._train_done.set()



class SequentialOffPolicyAgent(OffPolicyAgent):
  """Runs the environment and trains the model sequentially"""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Use a thread in order to exit cleanly on KeyboardInterrupt
    train_thread  = threading.Thread(name='train_thread', target=self._train)
    self.threads  = [train_thread]


  def _train(self):

    obs = self.reset()

    for t in range (self.start_step, self.stop_step+1):
      if self._terminate:
        break

      # Get an action to run
      if self.learn_started:
        action = self._action_train(obs, t)

      # Choose random action if learning has not started
      else:
        action = self.env.action_space.sample()

      # Increment the TF timestep variable
      self.sess.run(self.t_train_inc)

      # Run action
      next_obs, reward, done, _ = self.env.step(action)

      # Store the effect of the action taken upon obs
      self.replay_buf.store(obs, action, reward, done)

      # Reset the environment if end of episode
      if done:
        next_obs = self.reset()
      obs = next_obs

      # Train the model
      if (t >= self.warm_up and t % self.train_freq == 0):

        self.learn_started = True

        # Run a training step
        run_summary = t % self.log_freq == 0
        self._run_train_step(t, run_summary=run_summary)

      # Save data
      if self.save_freq > 0 and t % self.save_freq == 0:
        self.save()

      # Log data
      self._log_stats(t)

      # Stop and run evaluation procedure
      if self.eval_len > 0 and t % self.eval_freq == 0:
        self.eval()
        # Reset the environment on return
        obs = self.reset()


  def _wait_act_chosen(self):
    return
