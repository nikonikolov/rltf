import threading

from rltf.agents import LoggingAgent
from rltf.agents import ThreadedAgent


class BaseQlearnAgent(LoggingAgent, ThreadedAgent):
  """The base class for a Q-learning based agents. Assumes usage of target network and replay buffer.
  Runs training and evaluation in python Threads for safe exit when Ctrl+C is pressed
  """

  def __init__(self,
               warm_up,
               train_period,
               target_update_period,
               stop_step,
               *args,
               save_buf=True,
               **kwargs):

    """
    Args:
      warm_up: int. Number of random steps before training starts
      train_period: int. How many environment actions to take between every 2 learning steps
      target_update_period: Period in number of agent steps at which to update the target net
      stop_step: int. Training step at which learning stops
      save_buf: bool. If True, save the buffer during calls to `self.save()`. Can also be disabled
        by setting the 'RLTFBUF' environment variable to `/dev/null`.
    """
    super().__init__(*args, **kwargs)

    # Training data
    self.warm_up        = warm_up       # Step from which training starts
    self.learn_started  = False         # Bool: Indicates if learning has started or not
    self.train_period   = train_period  # How often to run a training step
    self.stop_step      = stop_step     # Step at which training stops

    self.target_update_period = target_update_period  # Period at which target network is updated
    self.replay_buf = None

    self.threads    = []
    self.save_buf   = save_buf


  def _train(self):
    self._run_threads(self.threads)


  def _restore(self):
    self.replay_buf.restore(self.model_dir)


  def _save(self):
    super()._save()
    if self.save_buf:
      self.replay_buf.save(self.model_dir)


  def _save_allowed(self):
    return self.learn_started


  def _run_train_step(self, t):
    # Compose feed_dict
    batch       = self.replay_buf.sample(self.batch_size)
    feed_dict   = self._get_feed_dict(batch, t)

    # Wait for synchronization if necessary
    self._wait_act_chosen()

    # Run a training step
    self.sess.run(self.model.train_op, feed_dict=feed_dict)

    # Update target network
    if t % self.target_update_period == 0:
      self.sess.run(self.model.update_target)

    # Run the summary op to log the changes from the update if necessary
    self._run_summary_op(t, feed_dict)


  def _run_summary_op(self, t, feed_dict):
    # Remember this is called only each training period
    # Make sure to run the summary right before t gets to a log_period so as to make sure
    # that the summary will be updated on time
    run_summary = t % self.log_period + self.train_period >= self.log_period

    if run_summary:
      self.summary = self.sess.run(self.summary_op, feed_dict=feed_dict)


  def _wait_act_chosen(self):
    raise NotImplementedError()


  def _action_train(self, state, t):
    """Return action selected by the agent for a training step
    Args:
      state: np.array. Current state
      t: int. Current timestep
    """
    raise NotImplementedError()



class QlearnAgent(BaseQlearnAgent):
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

    # env_thread    = threading.Thread(name='env_thread', target=self._run_env)
    # nn_thread     = threading.Thread(name='net_thread', target=self._train_model)
    env_thread    = threading.Thread(name='env_thread', target=self._thread, args=[self._run_env])
    nn_thread     = threading.Thread(name='net_thread', target=self._thread, args=[self._train_model])
    self.threads  = [nn_thread, env_thread]


  def _run_env(self):
    """Thread for running the environment. Must call `self._wait_train_done()`
    before selcting an action (by running the model). This ensures that the
    `self._train_model()` thread has finished the training step. After action
    is selected, it must call `self._signal_act_chosen()` to allow
    `self._train_model()` thread to start a new training step
    """

    obs = self.reset()

    for t in range(self.agent_step+1, self.stop_step+1):
      if self._terminate:
        self._signal_act_chosen()
        break

      # Get an action to run
      if self.learn_started:
        action = self._action_train(obs, t)

      # Choose random action if learning has not started
      else:
        action = self.env_train.action_space.sample()

      # Signal to net_thread that action is chosen
      self._signal_act_chosen()

      # Run action
      next_obs, reward, done, _ = self.env_train.step(action)

      # Store the effect of the action taken upon obs
      self.replay_buf.store(obs, action, reward, done)

      # Wait until net_thread is done
      self._wait_train_done()

      # Reset the environment if end of episode
      if done:
        next_obs = self.reset()
      obs = next_obs

      # Stop and run evaluation procedure
      if self.eval_len > 0 and t % self.eval_period == 0:
        self._eval_agent()

      # Update the agent step
      self.agent_step = t

      # Save **after** agent step is correct and completed
      if t % self.save_period == 0:
        self.save()


  def _train_model(self):
    """Thread for trianing the model. Must call `self._wait_act_chosen()`
    before trying to run a training step on the model. This ensures that the
    `self._run_env()` thread has finished selcting an action (by running the model).
    After training step is done, it must call `self._signal_train_done()` to allow
    `self._run_env()` thread to select a new action
    """

    for t in range(self.agent_step+1, self.stop_step+1):
      if self._terminate:
        self._signal_train_done()
        break

      if (t >= self.warm_up and t % self.train_period == 0):

        self.learn_started = True

        # Run a training step
        self._run_train_step(t)

      else:
        # Synchronize
        self.replay_buf.wait_stored()
        self.replay_buf.signal_sampled()
        self._wait_act_chosen()

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



class SequentialQlearnAgent(BaseQlearnAgent):
  """Runs the environment and trains the model sequentially. End result is the same as QlearnAgent.
  Provided for verifying correctness. if needed."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Use a thread in order to exit cleanly on KeyboardInterrupt
    # train_thread  = threading.Thread(name='train_thread', target=self._train_model)
    self.threads  = [threading.Thread(name='train_thread', target=self._thread, args=[self._train_model])]


  def _train_model(self):

    obs = self.reset()

    for t in range(self.agent_step+1, self.stop_step+1):
      if self._terminate:
        break

      # Get an action to run
      if self.learn_started:
        action = self._action_train(obs, t)

      # Choose random action if learning has not started
      else:
        action = self.env_train.action_space.sample()

      # Run action
      next_obs, reward, done, _ = self.env_train.step(action)

      # Store the effect of the action taken upon obs
      self.replay_buf.store(obs, action, reward, done)

      # Reset the environment if end of episode
      if done:
        next_obs = self.reset()
      obs = next_obs

      # Train the model
      if (t >= self.warm_up and t % self.train_period == 0):

        self.learn_started = True

        # Run a training step
        self._run_train_step(t)

      # Stop and run evaluation procedure
      if self.eval_len > 0 and t % self.eval_period == 0:
        self._eval_agent()

      # Update the agent step
      self.agent_step = t

      # Save **after** agent step is correct and completed
      if t % self.save_period == 0:
        self.save()


  def _wait_act_chosen(self):
    return
