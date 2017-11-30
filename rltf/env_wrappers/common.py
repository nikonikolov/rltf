# from rltf.env_wrappers.wrappers import ResizeFrame
# from rltf.env_wrappers.wrappers import RepeatAndStackImage

def wrap_deepmind_ddpg(env):
  if len(env.observation_space.shape) == 3:
    # env = ResizeFrame(env)
    # env = RepeatAndStackImage(env)
    raise NotImplementedError()

  return env
