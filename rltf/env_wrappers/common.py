# from rltf.env_wrappers.wrappers import ResizeFrame
# from rltf.env_wrappers.wrappers import RepeatAndStackImage
from rltf.env_wrappers.wrappers import ScaleReward

def wrap_deepmind_ddpg(env, scale=1.0):
  if scale != 1.0:
    env = ScaleReward(env)
  if len(env.observation_space.shape) == 3:
    # env = ResizeFrame(env)
    # env = RepeatAndStackImage(env)
    raise NotImplementedError()

  return env
