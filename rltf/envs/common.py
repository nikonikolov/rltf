# from rltf.envs.wrappers import ResizeFrame
# from rltf.envs.wrappers import RepeatAndStackImage
from rltf.envs.wrappers import ClipAction
from rltf.envs.wrappers import MaxEpisodeLen
from rltf.envs.wrappers import NormalizeAction
from rltf.envs.wrappers import ScaleReward

def wrap_deepmind_ddpg(env, rew_scale=1.0, max_ep_len=None):
  env = NormalizeAction(env)
  env = ClipAction(env)
  if rew_scale != 1.0:
    env = ScaleReward(env, rew_scale)
  if len(env.observation_space.shape) == 3:
    # env = ResizeFrame(env)
    # env = RepeatAndStackImage(env)
    raise NotImplementedError()
  if max_ep_len is not None:
    env = MaxEpisodeLen(env, max_ep_len)

  return env
