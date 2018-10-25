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
from gym.utils    import closer
from gym.error    import ResetNeeded
from gym          import __version__ as GYM_VERSION
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from rltf.monitoring.stats import StatsRecorder
from rltf.monitoring.vplot import VideoPlotter


logger = logging.getLogger(__name__)


class Monitor(Wrapper):
  """Custom implementation of a Monitor class which has more functionality than gym.Monitor.
  Supports logging training and evaluation statistics in real time, video recording, clear separation
  between train and evaluation mode, saving statistrics to disk in numpy format.

  NOTE: For total safety, this wrapper must be applied directly on top of the environment, without
  any other wrappers in between. Otherwise, the reported statistics might be incorrect.

  Based on `gym/gym/wrappers/monitor.py`
  """

  def __init__(self, env, log_dir, video_callable=None, mode='t'):
    """
    Args:
      log_dir: str. The directory where to save the monitor videos and stats
      video_callable: function or False. False disables video recording. If function is provided, it
        has to take the number of the episode and return True/False if a video should be recorded.
        If `None`, every 1000th episode is recorded
      mode: str. Either 't' (train) or 'e' (eval) for the mode in which to start the monitor
    """

    self._detect_wrapped_env(env)

    # Wrap the enviroment for plotting in the video. Disabled by default
    env = VideoPlotter(env)

    super().__init__(env)

    self.videos         = []      # List of files for the recorded videos and their manifests
    self.log_dir        = log_dir
    self.done           = None
    self.env_started    = False
    self.env_id         = self._get_env_id()
    self.video_callable = self._get_video_callable(video_callable)

    self.stats_recorder = StatsRecorder(os.path.join(self.log_dir, "data"), mode)
    self.video_recorder = None

    # Create the monitor directory
    self._make_log_dir()

    # Attach StatsRecorder methods
    self.define_log_info  = self.stats_recorder.define_log_info
    self.log_stats        = self.stats_recorder.log_stats
    self.start_eval_run   = self.stats_recorder.start_eval_run
    self.save             = self.stats_recorder.save
    self.end_eval_run     = self.stats_recorder.end_eval_run


  def _get_video_callable(self, video_callable):
    # Set the video recording schedule
    if video_callable is None:
      video_callable = lambda e_id: e_id % 1000 == 0
    elif video_callable is False:
      video_callable = lambda e_id: False
    elif not callable(video_callable):
      raise ValueError("You must provide a function, None, or False for 'video_callable', "
                       "not {}: {}".format(type(video_callable), video_callable))
    return video_callable


  def _get_env_id(self):
    if self.env.spec is None:
      logger.warning("Trying to monitor an environment which has no 'spec' set."
                     "This usually means you did not create it via 'gym.make',"
                     "and is recommended only for advanced users.")
      return '(unknown)'
    return self.env.spec.id


  def _make_log_dir(self):
    if not os.path.exists(self.log_dir):
      logger.info('Creating monitor directory %s', self.log_dir)
      os.makedirs(self.log_dir)


  def _detect_wrapped_env(self, env):
    if isinstance(env, Wrapper):
      if not isinstance(env, TimeLimit):
        logger.warning("Trying to monitor the environment %s wrapped with %s. Reported statistics might "
                       "be incorrect", env.spec.id, type(env))


  def step(self, action):
    self._before_step(action)
    obs, reward, done, info = self.env.step(action)
    done = self._after_step(obs, reward, done, info)
    return obs, reward, done, info


  def reset(self, **kwargs):
    # First reset stats for correct episode_id
    self.stats_recorder.reset()
    # obs = self.env.reset(**kwargs)
    obs = self.env.reset(enabled=self.video_callable(self.episode_id), mode=self.mode, **kwargs)
    self._after_reset(obs)
    return obs


  def close(self):
    """Flush all monitor data to disk and close any open rending windows."""

    # First save all the data
    # self.save()

    # Close stats and video recorders
    # self.stats_recorder.close()
    if self.video_recorder is not None:
      self._close_video_recorder()

    # logger.info("Monitor successfully closed and saved at %s", self.log_dir)
    # logger.info("Monitor successfully closed")


  def _before_step(self, action):
    if self.done:
      raise error.ResetNeeded("Trying to step environment which is currently done."
        "While the monitor is active for {}, you cannot step beyond the end of an episode."
        "Call 'env.reset()' to start the next episode.".format(self.env_id))

    if not self.env_started:
      raise error.ResetNeeded("Trying to step an environment before reset."
        "While the monitor is active for {}, you must call 'env.reset()'"
        "before taking an initial step.".format(self.env_id))


  def _after_step(self, obs, reward, done, info):
    # Record stats and video
    self.stats_recorder.after_step(obs, reward, done, info)
    self.video_recorder.capture_frame()
    self.done = done

    return done


  def _after_reset(self, obs):
    self.env_started = True
    self.done = False

    # Start new video recording
    self.reset_video_recorder()


  def reset_video_recorder(self):
    """Close the current video recorder and open a new one. Automatically stops the
    current video and starts a new one
    """

    # Close any existing video recorder
    if self.video_recorder:
      self._close_video_recorder()

    ep_id = self.episode_id
    mode  = self.stats_recorder.mode
    video_file = "openaigym_video_{}_episode_{:06}".format("train" if mode == 't' else "eval", ep_id)
    video_file = os.path.join(self.log_dir, video_file)

    # Start recording the next video
    self.video_recorder = VideoRecorder(
      env=self.env,
      base_path=video_file,
      metadata={'episode_id': ep_id},
      enabled=self.video_callable(ep_id),
    )
    self.video_recorder.capture_frame()


  def _close_video_recorder(self):
    # Close the recorder
    self.video_recorder.close()

    # If the video was recorded and successfully written, then remember its path
    if self.video_recorder.functional:
      self.videos.append((self.video_recorder.path, self.video_recorder.metadata_path))


  def __del__(self):
    # Make sure we've closed up shop when garbage collecting
    self.close()


  def conf_video_plots(self, **kwargs):
    self.env.conf_plots(**kwargs)


  def __getattr__(self, attr):
    if attr in ["mode", "episode_id", "total_steps", "mean_ep_rew", "eval_score",
      "episode_rewards", "episode_lens",]:
      return getattr(self.stats_recorder, attr)
    raise AttributeError("%r object has no attribute %r" % (self.__class__, attr))
