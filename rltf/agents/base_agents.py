import logging
from abc import ABCMeta, abstractmethod

import tensorflow as tf

from rltf.agents import Agent


logger = logging.getLogger(__name__)


class ThreadedAgent(Agent):
  """Abstract agent which can run the main process in several python threads.
  Allows for a standardized way of exiting cleanly, without losing any data at Ctrl+C"""

  def _run_threads(self, threads):
    """Run `threads`
    Args:
      threads: list of threads to start and join
    """
    # Start threads
    for t in threads:
      t.start()
    # Wait for threads
    for t in threads:
      t.join()


  def _thread(self, f):
    """Share the default graph over threads"""
    assert self.sess is not None
    with self.sess.graph.as_default():
      f()



class LoggingAgent(Agent, metaclass=ABCMeta):
  """Abstract Agent which takes care of logging training and evaluation progress to stdout
  and TensorBoard. Also takes care of saving data to disk and restoring it"""

  def __init__(self, *args, log_period=50000, video_period=1000, plot_video=False, **kwargs):
    """
    Args:
      log_period: int. Add TensorBoard summary and print progress every log_period agent steps
      video_period: int. Period for recording episode videos (in number of episodes). If <=0,
        no recordings will be made
      plot_video: bool. If True, plots of some of the model tensor values will be included in video
        recordings by the monitor. Values appear together with the corresponding state
    """
    super().__init__(*args, **kwargs)

    self.log_period   = log_period
    self.video_period = video_period
    self.plot_video   = plot_video
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

    if self.plot_video:
      # Enable plotting tensors in the recorded videos
      self.env_train.monitor.enable_video_plots(self.model.name)
      self.env_eval.monitor.enable_video_plots(self.model.name)

    # No need for deactivating. It is deactivated by default
    # else:
    #   # If plots not enabled, make sure no tensors are run needlessly
    #   self.model.plot_conf.deactivate_train_plots()
    #   self.model.plot_conf.deactivate_eval_plots()


  def _fetch_summary(self):
    byte_summary = self.summary
    summary = tf.Summary()
    if byte_summary is not None:
      summary.ParseFromString(byte_summary)
    # Pass the real current training step
    self._append_summary(summary, self.agent_step+1)

    return summary


  def _save(self):
    # Save the monitor statistics
    self.env_train.monitor.save()
    self.env_eval.monitor.save()


  @abstractmethod
  def _run_summary_op(self, t, feed_dict):
    """Run the summary op and save the result in self.summary
    NOTE:
      - For significant computation efficiency, summaries should be run only each log_period,
        otherwise the data is thrown away and causes unnecessary computations
      - Careful with implementation. Remember that the summary is actually fetched by the
        environment monitor, at every log_period, **during the call to env.step()**. Importantly,
        this means that the summary has to be run **before** the corresponding call to env.step()
    Args:
      t: int. Current time step
      feed_dict: dict. feed_dict to feed to sess.run
    """
    pass


  def _append_log_spec(self):
    """To be overriden by the subclass
    Returns:
      List of tuples `(name, format, lambda)` with information of custom subclass
      parameters to log during training. `name`: `str`, the name of the reported
      value. `modifier`: `str`, the type modifier for printing the value.
      `lambda`: A function that takes the current timestep as argument and
      returns the value to be printed.
    """
    return []


  #pylint: disable=unused-argument
  def _append_summary(self, summary, t):
    """To be overriden by the subclass.
    Append the tf.Summary that will be written to disk at timestep t with custom data.
    Used only in train mode. The resulting summary is passed to rltf.Monitor for saving.
    Args:
      summary: tf.Summary. The summary to append
      t: int. Current time step
    """
    return
