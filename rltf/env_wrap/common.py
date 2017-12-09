# from rltf.env_wrap.wrappers import ResizeFrame
# from rltf.env_wrap.wrappers import RepeatAndStackImage
from rltf.env_wrap.wrappers import ScaleReward

def wrap_deepmind_ddpg(env, rew_scale=1.0):
  if rew_scale != 1.0:
    env = ScaleReward(env, rew_scale)
  if len(env.observation_space.shape) == 3:
    # env = ResizeFrame(env)
    # env = RepeatAndStackImage(env)
    raise NotImplementedError()

  return env
