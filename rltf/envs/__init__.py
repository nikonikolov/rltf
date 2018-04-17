from rltf.envs.atari  import wrap_deepmind_atari
from rltf.envs.common import wrap_deepmind_ddpg
from rltf.envs.common import wrap_dqn

from gym.envs.registration import register

register(
    id='NoiseGrid-v0',
    entry_point='rltf.envs.noise_grid:NoiseGrid',
    # max_episode_steps=1000,
)
