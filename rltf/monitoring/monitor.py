import logging
import os

from gym          import Wrapper
from gym.wrappers import TimeLimit
from gym.error    import ResetNeeded
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from rltf.monitoring.stats import StatsRecorder
from rltf.monitoring.vplot import VideoPlotter
from rltf.envs             import MaxEpisodeLen


logger = logging.getLogger(__name__)


class Monitor(Wrapper):
  """A Monitor class which automatically tracks training or evaluation statistics and saves them to disk.
  Supports logging statistics to stdout and to a file in real time, video recording, adding plots to
  the video, saving TensorBoard summaries, printing selected entries of the summary to stdout,
  saving statistics to disk in numpy format so they can be easily plotted later.

  NOTE: This wrapper must be applied directly on top of the environment, without any other wrappers
  between the agent and the monitor. Otherwise, the reported statistics might be incorrect. The monitor
  gets the unwrapped environment or its TimeLimit wrapper and attaches to the corresponding `step()`,
  `reset()` and `render()` functions in order to be able to passively collect statistics (the actual
  functions are left intact).

  The monitor supports only a single mode of operation - either training or evaluation. This means
  that the agent should use different environment instances for training and evaluation.
  Statistics are tracked by StatsRecorder and it autmatically outputs basic data such as
  current agent and environment steps (not always equal), number of episodes, the mean and the std
  of the reward and of the episode length over the last 100 episodes, the best score so far, etc.
  Additional data can also easily be included in the output logs or the TensorBoard summary.

  Any TensorBoard scalar summary whose tag starts with "stdout/" and is added to the default summary list
  (tf.GraphKeys.SUMMARIES) is automatically included in the stdout logs (and removed from TensorBoard).
  However, the user needs to provide a method for fetching the most recent result of running all summary
  ops in a tf.Session. Be careful with running the environment in a thread, as the TensorFlow graph
  is not automatically shared between threads.

  Assumptions:
    - The environment has no other wrapper other than gym.TimeLimit or rltf.envs.MaxEpisodeLen
      that modifies the true episode length
    - The TF graph is available in the thread running `env.step()`
  """

  def __init__(self, env, log_dir, mode, log_period=None, video_spec=None, epochs=False, eval_period=None):
    """
    Args:
      log_dir: str. The directory where to save the monitor videos and stats
      log_period: int. The period for logging statistic to stdout and to TensorBoard
      mode: str. Either 't' (train) or 'e' (eval) for the mode in which to start the monitor
      video_spec: lambda, int, False or None. Specifies how often to record episodes.
        - If lambda, it must take the episode number and return True/False if a video should be recorded.
        - `int` specifies a period in episodes
        - `False`, disables video recording
        - If `None`, every 1000th episode is recorded
      epochs: If True, statistics are logged in terms of epochs, not agent steps
      eval_period: int. Required only in evaluation mode. Needed to compute the correct logging step.
    """

    assert mode in ['t', 'e']

    super().__init__(env)

    log_dir   = os.path.join(log_dir, "monitor")
    video_dir = os.path.join(log_dir, "videos")

    # Member data
    self.video_dir    = video_dir
    self.log_dir      = log_dir
    self.enable_video = self._get_video_callable(video_spec)

    # Create the monitor directory
    self._make_log_dir()

    # Composition objects
    self.stats_recorder = StatsRecorder(log_dir, mode, log_period, epochs, eval_period)
    self.video_plotter  = VideoPlotter(self.env, mode=mode)
    self.video_recorder = None

    # Attach StatsRecorder agent methods
    self._before_agent_step   = self.stats_recorder.before_agent_step
    self._after_agent_step    = self.stats_recorder.after_agent_step
    self._before_agent_reset  = self.stats_recorder.before_agent_reset
    self._after_agent_reset   = self.stats_recorder.after_agent_reset

    # Find TimeLimit wrapper and attach step(), reset() and render()
    self.base_env         = None
    self.base_env_step    = None
    self.base_env_reset   = None
    self.base_env_render  = None
    self._attach_env_methods()

    # Attach member methods
    self.enable_video_plots = self.video_plotter.activate
    self.set_stdout_logs    = self.stats_recorder.set_stdout_logs
    self.set_summary_getter = self.stats_recorder.set_summary_getter
    self.save               = self.stats_recorder.save
    self.log_stats          = self.stats_recorder.log_stats


  def _attach_env_methods(self):
    """Attach self._env_step to the TimeLimit wrapper of the environment or if not present, to
    the unwrapped environment. Attach to TimeLimit in order to track the true env reset signals"""

    # Get the TimeLimit Wrapper or the unwrapped env
    currentenv = self.env
    while True:
      if isinstance(currentenv, (TimeLimit, MaxEpisodeLen)):
        break
      elif isinstance(currentenv, Wrapper):
        currentenv = currentenv.env
      else:
        break

    # Attach the **env** step function
    self.base_env         = currentenv
    self.base_env_step    = self.base_env.step
    self.base_env_reset   = self.base_env.reset
    self.base_env_render  = self.base_env.render
    self.base_env.step    = self._env_step
    self.base_env.reset   = self._env_reset
    self.base_env.render  = self._env_render


  def step(self, action):
    """Corresponds to the step function executed by the agent"""
    return self._agent_step(action)


  def reset(self, **kwargs):
    self._before_agent_reset()
    obs = self.env.reset(**kwargs)
    self._after_agent_reset()
    return obs


  def _agent_step(self, action):
    self._before_agent_step(action)
    obs, reward, done, info = self.env.step(action)
    self._after_agent_step(obs, reward, done, info)
    return obs, reward, done, info


  def _env_step(self, action):
    """Corresponds to the step function of the TimeLimit wrapper or the unwrapped environment"""
    self._before_env_step(action)
    # Call the actual env.step function
    obs, reward, done, info = self.base_env_step(action)
    self._after_env_step(obs, reward, done, info)
    return obs, reward, done, info


  def _before_env_step(self, action):
    # Do not execute if the environment was not stepped or reset from this monitor
    if not self._active:
      # Remember that env was not stepped via the monitor and require reset next time
      self.stats_recorder.env_done = None
      return

    if self.done is None:
      raise ResetNeeded("Trying to step environment {}, before calling 'env.reset()'.".format(self.env_id))

    if self.done:
      raise ResetNeeded("Trying to step environment {}, which is done. You cannot step beyond the "
        "end of an episode. Call 'env.reset()' to start the next episode.".format(self.env_id))


  def _after_env_step(self, obs, reward, done, info):
    # Do not execute if the environment was not stepped or reset from this monitor
    if not self._active:
      return

    # Record stats and video
    self.video_recorder.capture_frame()
    self.stats_recorder.after_env_step(obs, reward, done, info)


  def _env_reset(self, **kwargs):
    obs = self.base_env_reset(**kwargs)

    # Do not execute if the environment was not stepped or reset from this monitor
    if not self._active:
      return obs

    # First reset stats for correct episode_id
    self.stats_recorder.env_reset()
    # Reset the video plotter next so it can prepare
    self.video_plotter.reset(enabled=self.enable_video(self.episode_id))
    # Start new video recording
    self._reset_video_recorder()

    return obs


  def _env_render(self, mode):
    obs = self.base_env_render(mode)
    # Execute only if the environment was stepped or reset from this monitor
    if self._active:
      obs = self.video_plotter.render(obs, mode)
    return obs


  def close(self):
    """Flush all monitor data to disk and close any open rending windows."""

    # Close stats recorder
    self.stats_recorder.close()

    # Close video recorder
    if self.video_recorder is not None:
      self._close_video_recorder()

    # Close the environment
    if self.env:
      return self.env.close()

    # logger.info("Monitor successfully closed and saved at %s", self.log_dir)
    # logger.info("Monitor successfully closed")


  def _reset_video_recorder(self):
    """Close the current video recorder and open a new one. Automatically stops the
    current video and starts a new one
    """

    # Close any existing video recorder
    if self.video_recorder:
      self._close_video_recorder()

    ep_id = self.episode_id
    mode  = self.stats_recorder.mode
    video_file = "{}_video_episode_{:06}".format("train" if mode == 't' else "eval", ep_id)
    video_file = os.path.join(self.video_dir, video_file)

    # Start recording the next video
    self.video_recorder = VideoRecorder(
      env=self.env,
      base_path=video_file,
      metadata={'episode_id': ep_id},
      enabled=self.enable_video(ep_id),
    )
    self.video_recorder.capture_frame()


  def _close_video_recorder(self):
    # Close the recorder
    self.video_recorder.close()


  def __del__(self):
    # Make sure we've closed up shop when garbage collecting
    self.close()


  def _make_log_dir(self):
    if not os.path.exists(self.log_dir):
      logger.info('Creating monitor directory %s', self.log_dir)
      os.makedirs(self.log_dir)

    if not os.path.exists(self.video_dir):
      os.makedirs(self.video_dir)


  def __getattr__(self, attr):
    if attr in ["mode", "episode_rews", "episode_lens"]:
      return getattr(self.stats_recorder, attr)
    raise AttributeError("%r object has no attribute %r" % (self.__class__, attr))


  @property
  def _active(self):
    """Track whether env.step and env.reset() were executed via this monitor"""
    return self.stats_recorder.active


  @property
  def monitor(self):
    return self


  @property
  def done(self):
    return self.stats_recorder.env_done


  @property
  def episode_id(self):
    return self.stats_recorder.env_eps


  @property
  def env_id(self):
    if self.env.spec is None:
      logger.warning("Trying to monitor an environment which has no 'spec' set. "
                     "This usually means you did not create it via 'gym.make', "
                     "and is recommended only for advanced users.")
      return '(unknown)'
    return self.env.spec.id


  @staticmethod
  def _get_video_callable(video_spec):
    # Set the video recording schedule
    if video_spec is None:
      video_spec = lambda e_id: e_id % 1000 == 0
    elif isinstance(video_spec, int):
      if video_spec > 0:
        period     = video_spec
        video_spec = lambda e_id: e_id % period == 0
      else:
        video_spec = lambda e_id: False
    elif video_spec is False:
      video_spec = lambda e_id: False
    elif not callable(video_spec):
      raise ValueError("You must provide a function, int, False, or None for 'video_spec', "
                       "not {}: {}".format(type(video_spec), video_spec))
    return video_spec
