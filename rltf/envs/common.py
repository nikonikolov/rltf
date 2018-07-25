# from rltf.envs.wrappers import ResizeFrame
# from rltf.envs.wrappers import RepeatAndStackImage
from rltf.envs.wrappers import ClipAction
from rltf.envs.wrappers import NormalizeAction
from rltf.envs.wrappers import ScaleReward
from rltf.envs.atari    import wrap_deepmind_atari
from rltf.envs.atari    import ClippedRewardsWrapper


def wrap_deepmind_ddpg(env, mode, rew_scale=1.0):
  env = NormalizeAction(env)
  env = ClipAction(env)
  if mode == 't' and rew_scale != 1.0:
    env = ScaleReward(env, rew_scale)
  if len(env.observation_space.shape) == 3:
    # env = ResizeFrame(env)
    # env = RepeatAndStackImage(env)
    raise NotImplementedError()
  return env


def _wrap_nonimg_dqn(env, mode):
  if mode == 't':
    env = ClippedRewardsWrapper(env)
  return env


def wrap_dqn(env, mode, **kwargs):
  if len(env.observation_space.shape) == 3:
    return wrap_deepmind_atari(env, mode, **kwargs)
  else:
    return _wrap_nonimg_dqn(env, mode, **kwargs)
