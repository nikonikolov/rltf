import logging
import tensorflow as tf

from rltf.agents import Agent


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

  def __init__(self, *args, log_period=10000, plots_layout=None, **kwargs):
    """
    Args:
      log_period: int. Add TensorBoard summary and print progress every log_period agent steps
      plots_layout: dict or None. Used to configure the layout for video plots
    """
    super().__init__(*args, **kwargs)

    self.log_period   = log_period
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
