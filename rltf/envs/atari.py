# Partially based on https://github.com/openai/baselines under the following license:
#
# The MIT License
#
# Copyright (c) 2017 OpenAI (http://openai.com)
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
from collections import deque

import cv2
import gym
import numpy as np

# from rltf.utils import seeding

logger = logging.getLogger(__name__)


class NoopResetEnv(gym.Wrapper):
  """Execute a random number of no-ops on reset. No-op is assumed to be action 0."""

  def __init__(self, env, noop_max=30):
    super().__init__(env)
    self.noop_max = noop_max
    assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    # self.prng = seeding.get_prng()

  #pylint: disable=method-hidden
  def step(self, action):
    return self.env.step(action)

  #pylint: disable=method-hidden
  def reset(self, **kwargs):
    """Do no-op action for a number of steps in [1, noop_max]."""
    self.env.reset(**kwargs)
    # noops = self.prng.randint(1, self.noop_max + 1)
    noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
    for _ in range(noops):
      obs, _, done, _ = self.env.step(0)
      if done:
        # logger.warning("Environment reset during initial NOOPs")
        obs = self.env.reset(**kwargs)
    return obs


class FireResetEnv(gym.Wrapper):
  """Take action on reset for environments that are fixed until firing."""

  def __init__(self, env):
    super().__init__(env)
    assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
    assert len(env.unwrapped.get_action_meanings()) >= 3

  #pylint: disable=method-hidden
  def step(self, action):
    return self.env.step(action)

  #pylint: disable=method-hidden
  def reset(self, **kwargs):
    self.env.reset(**kwargs)
    obs, _, done, _ = self.env.step(1)
    if done:
      self.env.reset(**kwargs)
    obs, _, done, _ = self.env.step(2)
    if done:
      self.env.reset(**kwargs)
    return obs


class EpisodicLifeEnv(gym.Wrapper):
  """Make end-of-life == end-of-episode, but reset only on true game over.
  Done by DeepMind for the DQN and co. since it helps value estimation.
  NOTE: Must be applied only in TRAINING MODE. In evaluation mode end-of-life != end-of-episode
  """

  def __init__(self, env):
    super().__init__(env)
    self.lives = 0
    self.env_done = True

  #pylint: disable=method-hidden
  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    # Remember done as reported by env
    self.env_done = done
    # If a life was lost, broadcast done=True to upper levels: make
    # loss of life terminal. Then update lives to handle bonus lives
    lives = self.env.unwrapped.ale.lives()
    if lives < self.lives and lives > 0:
      # for Qbert somtimes we stay in lives == 0 condtion for a few frames
      # so its important to keep lives > 0, so that we only reset once
      # the environment advertises done.
      done = True
    self.lives = lives
    return obs, reward, done, info

  #pylint: disable=method-hidden
  def reset(self, **kwargs):
    """Reset only when lives are exhausted.
    This way all states are still reachable even though lives are episodic,
    and the learner need not know about any of this behind-the-scenes.
    """
    if self.env_done:
      obs = self.env.reset(**kwargs)
    else:
      # no-op step to advance from terminal/lost life state
      obs, _, done, _ = self.env.step(0)
      if done:
        obs = self.env.reset(**kwargs)
    self.lives = self.env.unwrapped.ale.lives()
    return obs


class MaxAndRepeatEnv(gym.Wrapper):
  """Repeat an action `repeat` times and return the element-wise max over the
  last 2 frames - done by DeepMind due to flickering objects.
  """

  def __init__(self, env, repeat=4):
    super().__init__(env)
    # most recent raw observations (for max pooling across time steps)
    # self._obs_buffer = deque(maxlen=2)
    self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
    self._repeat     = repeat
    assert self._repeat >= 1

  #pylint: disable=method-hidden
  def step(self, action):
    """Repeat action, sum reward, and max over last observations."""
    total_reward = 0.0
    for i in range(self._repeat):
      obs, reward, done, info = self.env.step(action)
      if i == self._repeat - 2: self._obs_buffer[0] = obs
      if i == self._repeat - 1: self._obs_buffer[1] = obs
      total_reward += reward
      if done:
        break

    # NOTE: The observation on the done=True doesn't matter - it is never used
    max_frame = self._obs_buffer.max(axis=0)

    return max_frame, total_reward, done, info

  #pylint: disable=method-hidden
  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
  """Warp frames to 84x84 as done in the Nature paper and later work."""

  def __init__(self, env):
    super().__init__(env)
    self.width = 84
    self.height = 84
    shape = (self.height, self.width, 1)
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

  def observation(self, observation):
    # COLOR_RGB2GRAY is eqivalent to Y channel
    # See CV docs at https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(observation, (self.width, self.height), interpolation=cv2.INTER_AREA)
    return observation[:, :, None]


class ClippedRewardsWrapper(gym.RewardWrapper):
  """Clip rewards in [-1, 1] range. NOTE: apply only for TRAINING MODE."""

  def reward(self, reward):
    return np.sign(reward)


class StackFrames(gym.Wrapper):
  """Stack the last k frames - done by DeepMind to infer object velocities."""

  def __init__(self, env, k=4):
    super().__init__(env)
    self.k        = k
    self.obs_buf  = deque([], maxlen=k)
    state_shape   = list(env.observation_space.shape)
    state_shape[-1] *= k
    dtype         = env.observation_space.dtype
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=state_shape, dtype=dtype)

  #pylint: disable=method-hidden
  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.obs_buf.append(obs)
    return self._obs(), reward, done, info

  #pylint: disable=method-hidden
  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    for _ in range(self.k):
      self.obs_buf.append(obs)
    return self._obs()

  def _obs(self):
    assert len(self.obs_buf) == self.k
    return np.concatenate(self.obs_buf, axis=-1)


def wrap_deepmind_atari(env, mode, stack=4):
  """Wraps an Atari environment to have the same settings as in the original DQN Nature paper by Deepmind.
  Args:
    env: gym.Env
    mode: str, either 't' or 'e'. Mode in which the environment will be run - train or eval
    stack: int. Number of frames to stack that comrise the agent observation
  Returns:
    The wrapped environment
  """
  if not isinstance(env.unwrapped, gym.envs.atari.AtariEnv):
    raise ValueError("Applying atari wrappers to the non-atari env {} is not allowed".format(env.spec.id))
  assert 'NoFrameskip' in env.spec.id
  assert mode in ['t', 'e']

  if mode == 't':
    env = EpisodicLifeEnv(env)
  env = NoopResetEnv(env, noop_max=30)
  env = MaxAndRepeatEnv(env, repeat=4)
  if 'FIRE' in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)
  env = WarpFrame(env)
  if mode == 't':
    env = ClippedRewardsWrapper(env)
  env = StackFrames(env, k=stack)
  return env
