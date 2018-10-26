# Partially based on https://github.com/openai/gym under the following license:
#
# The MIT License
#
# Copyright (c) 2016 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

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
  """Custom implementation of a Monitor class which has more functionality than gym.Monitor.
  Supports logging training and evaluation statistics in real time, video recording, clear separation
  between train and evaluation mode, saving statistrics to disk in numpy format.

  NOTE: For total safety, this wrapper must be applied directly on top of the environment, without
  any other wrappers in between. Otherwise, the reported statistics might be incorrect.

  Based on `gym/gym/wrappers/monitor.py`
  """

  def __init__(self, env, log_dir, log_period, mode, video_callable=None):
    """
    Args:
      log_dir: str. The directory where to save the monitor videos and stats
      log_period: int. The period for logging statistic to stdout and to TensorBoard
      video_callable: function or False. False disables video recording. If function is provided, it
        has to take the number of the episode and return True/False if a video should be recorded.
        If `None`, every 1000th episode is recorded
      mode: str. Either 't' (train) or 'e' (eval) for the mode in which to start the monitor
    """

    assert mode in ['t', 'e']

    super().__init__(env)

    log_dir   = os.path.join(log_dir, "monitor")
    stats_dir = os.path.join(log_dir, "data")
    video_dir = os.path.join(log_dir, "videos")

    # Member data
    self.video_dir    = video_dir
    self.log_dir      = log_dir
    self.enable_video = self._get_video_callable(video_callable)

    # Create the monitor directory
    self._make_log_dir()

    # Composition objects
    self.stats_recorder = StatsRecorder(stats_dir, log_period, mode)
    self.video_plotter  = VideoPlotter(self.env)
    self.video_recorder = None

    # Attach StatsRecorder agent methods
    self._before_agent_step = self.stats_recorder.before_agent_step
    self._after_agent_step  = self.stats_recorder.after_agent_step
    self._agent_reset       = self.stats_recorder.agent_reset

    # Find TimeLimit wrapper and attach step(), reset() and render()
    self.base_env         = None
    self.base_env_step    = None
    self.base_env_reset   = None
    self.base_env_render  = None
    self._attach_env_methods()

    # Attach member methods
    self.conf_video_plots = self.video_plotter.conf_plots
    self.set_stdout_logs  = self.stats_recorder.set_stdout_logs
    self.save             = self.stats_recorder.save


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
    self._agent_reset()
    return self.env.reset(**kwargs)


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
    if self.done is None:
      raise ResetNeeded("Trying to step environment {}, before calling 'env.reset()'.".format(self.env_id))

    if self.done:
      raise ResetNeeded("Trying to step environment {}, which is done. You cannot step beyond the "
        "end of an episode. Call 'env.reset()' to start the next episode.".format(self.env_id))


  def _after_env_step(self, obs, reward, done, info):
    # Record stats and video
    self.video_recorder.capture_frame()
    self.stats_recorder.after_env_step(obs, reward, done, info)


  def _env_reset(self, **kwargs):
    obs = self.base_env_reset(**kwargs)

    # First reset stats for correct episode_id
    self.stats_recorder.env_reset()
    # Reset the video plotter next so it can prepare
    self.video_plotter.reset(enabled=self.enable_video(self.episode_id), mode=self.mode)
    # Start new video recording
    self._reset_video_recorder()

    return obs


  def _env_render(self, mode):
    obs = self.base_env_render(mode)
    obs = self.video_plotter.render(obs, mode)
    return obs


  def close(self):
    """Flush all monitor data to disk and close any open rending windows."""

    # Close stats and video recorders
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
    video_file = "openaigym_video_{}_episode_{:06}".format("train" if mode == 't' else "eval", ep_id)
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
    if attr in ["mode", "mean_ep_rew", "episode_rews", "episode_lens", "eval_score",]:
      return getattr(self.stats_recorder, attr)
    raise AttributeError("%r object has no attribute %r" % (self.__class__, attr))


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
  def _get_video_callable(video_callable):
    # Set the video recording schedule
    if video_callable is None:
      video_callable = lambda e_id: e_id % 1000 == 0
    elif video_callable is False:
      video_callable = lambda e_id: False
    elif not callable(video_callable):
      raise ValueError("You must provide a function, None, or False for 'video_callable', "
                       "not {}: {}".format(type(video_callable), video_callable))
    return video_callable
