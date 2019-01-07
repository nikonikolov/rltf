# import cv2
import gym
import numpy as np


class ScaleReward(gym.RewardWrapper):
  """Scale rewards"""

  def __init__(self, env, scale):
    super().__init__(env)
    self.scale = scale

  def reward(self, reward):
    return self.scale * reward


class NormalizeAction(gym.ActionWrapper):
  """Receive actions in the range [-1, 1]"""

  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Box)
    super().__init__(env)
    assert np.any(self.action_space.high !=  float("inf"))
    assert np.any(self.action_space.low  != -float("inf"))
    self.act_mean = (self.action_space.high + self.action_space.low) / 2.0
    self.act_std  = (self.action_space.high - self.action_space.low) / 2.0
    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.action_space.shape, dtype=np.float32)

  def action(self, action):
    return self.act_std * action + self.act_mean

  def reverse_action(self, action):
    return (action - self.act_mean) / self.act_std


class ClipAction(gym.ActionWrapper):
  """Clip actions before playing them"""

  def __init__(self, env, low=-1.0, high=1.0):
    assert isinstance(env.action_space, gym.spaces.Box)
    super().__init__(env)
    self.high = high
    self.low  = low

  def action(self, action):
    return np.clip(action, self.low, self.high)

  def reverse_action(self, action):
    return action


class MaxEpisodeLen(gym.Wrapper):
  def __init__(self, env, max_episode_steps):
    """Limit episode length to max_steps"""
    super().__init__(env)
    self.max_steps  = max_episode_steps
    self.steps      = None

  #pylint: disable=method-hidden
  def step(self, action):
    self.steps += 1
    obs, reward, done, info = self.env.step(action)
    done = done or (self.steps >= self.max_steps)
    return obs, reward, done, info

  #pylint: disable=method-hidden
  def reset(self, **kwargs):
    self.steps = 0
    return self.env.reset(**kwargs)
