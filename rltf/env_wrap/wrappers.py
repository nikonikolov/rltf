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
    self.act_mean = (self.action_space.high + self.action_space.low) / 2.0
    self.act_std  = (self.action_space.high - self.action_space.low) / 2.0
    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.action_space.shape)

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



# class RepeatAndStackLowDim(gym.Wrapper):
#   """Repeat the action n times and stack the observations"""

#   def __init__(self, env, n):
#     """
#     Args:
#       n: int. Number of steps to repeat and stack
#     """
#     super().__init__(env)

#     obs_shape = list(env.observation_space.shape)

#     obs_shape = [n] + obs_shape
#     dtype = np.float32
#     low   = np.tile(env.observation_space.low[None:],   [n, 1])
#     high  = np.tile(env.observation_space.high[None:],  [n, 1])

#     self._obs_buf = np.zeros(obs_shape, dtype=nps.float32)
#     self._n       = n
#     self.observation_space = gym.spaces.Box(low=low, high=high, shape=obs_shape)


#   def step(self, action):
#     """Repeat action, sum reward, and stack observations. Note that if
#     episode terminates in the middle, the last observation is duplicated"""

#     total_reward = 0.0
#     done = False
#     for i in range(self._n):
#       if not done:
#         obs, reward, done, info = self.env.step(action)
#         total_reward += reward
#       self._obs_buf[i, :] = obs
#     # NOTE: DO NOTE RETURN self._obs_buf !!! This breaks encapsulation
#     # and will cause the object to later change the replay buffer unintentionally.
#     # Return a copy of self._obs_buf
#     return self._obs_buf, total_reward, done, info


# class RepeatAndStackImage(gym.Wrapper):
#   """Repeat the action n times and stack the observations"""

#   def __init__(self, env, n):
#     """
#     Args:
#       n: int. Number of steps to repeat and stack
#     """
#     super().__init__(env)

#     obs_shape = list(env.observation_space.shape)

#     self._nchannels = obs_shape[-1]

#     obs_shape[-1]   = n * self._nchannels

#     self._obs_buf = np.zeros(obs_shape, dtype=np.uint8)
#     self._n       = n
#     self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape)


#   def step(self, action):
#     """Repeat action, sum reward, and stack observations. Note that if
#     episode terminates in the middle, the last observation is duplicated"""

#     total_reward = 0.0
#     done = False
#     for i in range(self._n):
#       if not done:
#         obs, reward, done, info = self.env.step(action)
#         total_reward += reward
#       self._obs_buf[:, i*self._nchannels] = obs

#     return self._obs_buf, total_reward, done, info



# class ResizeFrame(gym.ObservationWrapper):
#   def __init__(self, env, width, height):
#     """Warp frames to 84x84 as done in the Nature paper and later work."""
#     super().__init__(env)
#     self.width = width
#     self.height = height
#     self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1))

#   def observation(self, frame):
#     # COLOR_RGB2GRAY is eqivalent to Y channel
#     # See CV docs at https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
#     return frame[:, :, None]

# def wrap_ddpg_image(env):
#   env = ResizeFrame(env)
#   env = RepeatAndStackImage(env)


# def wrap_ddpg_low_dim(env):
#   env = RepeatAndStackLowDim(env)

#   return env
