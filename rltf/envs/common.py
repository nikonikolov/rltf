# from rltf.envs.wrappers import ResizeFrame
# from rltf.envs.wrappers import RepeatAndStackImage
from rltf.envs.wrappers import ScaleReward
from rltf.envs.wrappers import NormalizeAction
from rltf.envs.wrappers import ClipAction

def wrap_deepmind_ddpg(env, rew_scale=1.0):
  env = NormalizeAction(env)
  env = ClipAction(env)
  if rew_scale != 1.0:
    env = ScaleReward(env, rew_scale)
  if len(env.observation_space.shape) == 3:
    # env = ResizeFrame(env)
    # env = RepeatAndStackImage(env)
    raise NotImplementedError()

  return env
