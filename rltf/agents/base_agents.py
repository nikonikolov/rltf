import logging
import threading
import tensorflow as tf

from rltf.agents.agent import Agent


logger = logging.getLogger(__name__)


class ThreadedAgent(Agent):
  """Abstract agent which can run the main process in several python threads.
  Allows for a standardized way of exiting cleanly, without losing any data at Ctrl+C"""

  def __init__(self, *args, confirm_kill=False, **kwargs):
    """
    Args:
      confirm_kill: bool. If True, you will be asked to confirm Ctrl+C
    """
    super().__init__(*args, **kwargs)

    self._terminate   = False         # Internal termination signal for the agent
    self.confirm_kill = confirm_kill


  def _run_threads(self, threads):
    """Safely run `threads` by sending an internal termination signal and joinining all threads
    before exiting. At Ctrl+C signal, optinally confirm that program must exit if confirm_kill is set.
    Args:
      threads: list of threads to start and join
    """

    # Start threads
    for t in threads:
      t.start()

    # Wait for threads
    stop = False
    while not stop:
      try:
        for t in threads:
          t.join()
        stop = True
      except KeyboardInterrupt:
        # Confirm the kill
        if not self._kill_confirmed():
          logger.info("CONTINUING EXECUTION")
          continue
        logger.info("EXITING")
        # Raise the internal termination signal
        self._terminate = True
        for t in threads:
          t.join()
        stop = True


  def _kill_confirmed(self):
    """Ask for confirmation for Ctrl+C or any additional termination signal
    Returns:
      `bool`. If True, kill. If False, continue
    """
    if self.confirm_kill:
      y = ''
      while True:
        y = input("Do you really want to exit? [y/n]. If you want to exit and save buffer, type 's'")
        if y not in ['y', 'n']:
          print("Response not recognized. Expected 'y' or 'n'.")
        else:
          break
      if y == 'n':
        return False
    return True


  def _thread(self, f):
    """Share the default graph over threads"""
    assert self.sess is not None
    with self.sess.graph.as_default():
      f()


  def _check_terminate(self):
    return self._terminate



class LoggingAgent(Agent):
  """Abstract Agent which takes care of logging training and evaluation progress to stdout
  and TensorBoard. Also takes care of saving data to disk and restoring it"""

  def __init__(self, *args, log_freq=10000, plots_layout=None, **kwargs):
    """
    Args:
      log_freq: int. Add TensorBoard summary and print progress every log_freq agent steps
      plots_layout: dict or None. Used to configure the layout for video plots
    """
    super().__init__(*args, **kwargs)

    self.log_freq     = log_freq
    self.plots_layout = plots_layout
    self.summary      = None    # The most recent summary
    self.summary_op   = None    # TF op that contains all summaries


  def build(self):
    # Build the actual graph
    super().build()

    # Create an Op for all summaries
    self.summary_op = tf.summary.merge_all()

    # Configure the monitors
    self._configure_monitors()


  def _configure_monitors(self):
    # Set stdout data to log during training
    self.env_train.monitor.set_stdout_logs(self._append_log_spec())

    # Set the function to fetch TensorBoard summaries during training
    self.env_train.monitor.set_summary_getter(self._fetch_summary)

    # Configure the plot layout for recorded videos
    if self.plots_layout is not None:
      self._plots_layout()
    else:
      self.model.clear_plot_tensors()


  def _plots_layout(self):
    self.env_train.monitor.conf_video_plots(layout=self.plots_layout, train_tensors=self.model.plot_train,
      eval_tensors=self.model.plot_eval, plot_data=self.model.plot_data)
    self.env_eval.monitor.conf_video_plots(layout=self.plots_layout, train_tensors=self.model.plot_train,
      eval_tensors=self.model.plot_eval, plot_data=self.model.plot_data)


  def _fetch_summary(self):
    byte_summary = self.summary
    summary = tf.Summary()
    if byte_summary is not None:
      summary.ParseFromString(byte_summary)
    # Pass the real current training step
    self._append_summary(summary, self.train_step+1)

    return summary


  def _save(self):
    # Save the monitor statistics
    self.env_train.monitor.save()
    self.env_eval.monitor.save()


  def _run_summary_op(self, t):
    """Return True if summary has to be run at this step, False otherwise.
    NOTE:
      - For significant computation efficiency, summaries should be run only each log_period,
        otherwise the data is thrown away and causes unnecessary computations
      - Careful with implementation. Remember that the summary is actually fetched by the
        environment monitor, at every log_period, **during the call to env.step()**. Importantly,
        this means that the summary has to be run **before** the corresponding call to env.step()
    Args:
      t: int. Current time step
    """
    raise NotImplementedError()


  def _append_log_spec(self):
    """
    Returns:
      List of tuples `(name, format, lambda)` with information of custom subclass
      parameters to log during training. `name`: `str`, the name of the reported
      value. `modifier`: `str`, the type modifier for printing the value.
      `lambda`: A function that takes the current timestep as argument and
      returns the value to be printed.
    """
    raise NotImplementedError()


  def _append_summary(self, summary, t):
    """Append the tf.Summary that will be written to disk at timestep t with custom data.
    Used only in train mode. The resulting summary is passed to rltf.Monitor for saving.
    Args:
      summary: tf.Summary. The summary to append
      t: int. Current time step
    """
    raise NotImplementedError()



class BaseQlearnAgent(LoggingAgent, ThreadedAgent):
  """The base class for a Q-learning based agents. Assumes usage of target network and replay buffer.
  Runs training and evaluation in python Threads for safe exit when Ctrl+C is pressed
  """

  def __init__(self, *args, save_buf=True, **kwargs):
    """
    Args:
      save_buf: bool. If True, save the buffer during calls to `self.save()`. Can also be disabled
        by setting the 'RLTFBUF' environment variable to `/dev/null`.
    """
    super().__init__(*args, **kwargs)

    self.update_target_freq = None
    self.replay_buf = None

    self.threads    = []
    self.save_buf   = save_buf


  def _train(self):
    self._run_threads(self.threads)


  def _eval(self):
    # Use a thread for clean exit on KeyboardInterrupt
    eval_thread = threading.Thread(name='eval_thread', target=self._thread, args=[self._run_eval])

    self._run_threads([eval_thread])


  def _restore(self):
    self.replay_buf.restore(self.model_dir)


  def _save(self):
    super()._save()
    if self.save_buf:
      self.replay_buf.save(self.model_dir)


  def _run_train_step(self, t):
    # Compose feed_dict
    batch       = self.replay_buf.sample(self.batch_size)
    feed_dict   = self._get_feed_dict(batch, t)
    run_summary = self._run_summary_op(t)

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


  def _run_summary_op(self, t):
    # Remember this is called only each training period
    # Make sure to run the summary right before t gets to a log_period so as to make sure
    # that the summary will be updated on time
    return t % self.log_freq + self.train_freq >= self.log_freq


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

    for t in range(self.train_step+1, self.stop_step+1):
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
      if self.eval_len > 0 and t % self.eval_freq == 0:
        self._eval_agent()

      # Update the train step
      self.train_step = t

      # Save **after** train step is correct and completed
      if self.save_freq > 0 and t % self.save_freq == 0:
        self.save()


  def _train_model(self):
    """Thread for trianing the model. Must call `self._wait_act_chosen()`
    before trying to run a training step on the model. This ensures that the
    `self._run_env()` thread has finished selcting an action (by running the model).
    After training step is done, it must call `self._signal_train_done()` to allow
    `self._run_env()` thread to select a new action
    """

    for t in range(self.train_step+1, self.stop_step+1):
      if self._terminate:
        self._signal_train_done()
        break

      if (t >= self.warm_up and t % self.train_freq == 0):

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

    for t in range(self.train_step+1, self.stop_step+1):
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
      if (t >= self.warm_up and t % self.train_freq == 0):

        self.learn_started = True

        # Run a training step
        self._run_train_step(t)

      # Stop and run evaluation procedure
      if self.eval_len > 0 and t % self.eval_freq == 0:
        self._eval_agent()

      # Update the train step
      self.train_step = t

      # Save **after** train step is correct and completed
      if self.save_freq > 0 and t % self.save_freq == 0:
        self.save()


  def _wait_act_chosen(self):
    return
