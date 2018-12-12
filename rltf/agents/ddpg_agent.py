import collections

import gym
import numpy as np

from rltf.agents      import QlearnAgent
from rltf.memory      import ReplayBuffer
from rltf.monitoring  import Monitor


class AgentDDPG(QlearnAgent):

  def __init__(self,
               env_maker,
               model,
               action_noise,
               memory_size=int(1e6),
               stack_frames=3,
               **agent_kwargs
              ):
    """
    Args:
      env_maker: callable. Function that takes the mode of an env and retruns a new environment instance
      model: rltf.models.Model. TF implementation of a model network
      action_noise: rltf.exploration.ExplorationNoise. Additive action space exploration noise
      memory_size: int. Size of the replay buffer
      stack_frames: int. How many frames comprise a single state.
      agent_kwargs: Keyword arguments that will be passed to the Agent base class
    """

    super().__init__(**agent_kwargs)

    self.env_train = Monitor(
                      env=env_maker('t'),
                      log_dir=self.model_dir,
                      mode='t',
                      log_period=self.log_period,
                      video_spec=self.video_period,
                    )

    self.env_eval  = Monitor(
                      env=env_maker('e'),
                      log_dir=self.model_dir,
                      mode='e',
                      log_period=self.eval_len,
                      video_spec=self.video_period,
                      eval_period=self.eval_period,
                    )

    self.action_noise = action_noise(self.env_train.action_space.shape)

    # Get environment specs
    obs_shape, obs_dtype, obs_len, act_shape = self._state_action_spec(stack_frames)

    # Initialize the model and the experience buffer
    self.model      = model(obs_shape=obs_shape, act_shape=act_shape, **self.model_kwargs)
    self.replay_buf = ReplayBuffer(memory_size, obs_shape, obs_dtype, act_shape, np.float32, obs_len)

    # Custom stats
    self.act_noise_stats = collections.deque([], maxlen=self.log_period)


  def _append_log_spec(self):
    t = self.log_period
    log_spec = [
      ( "mean_act_noise_mean (%d steps)"%t, "f", self._stats_act_noise_mean     ),
      ( "mean_act_noise_std  (%d steps)"%t, "f", self._stats_act_noise_std      ),
    ]
    return log_spec


  def _append_summary(self, summary, t):
    summary.value.add(tag="train/act_noise_mean", simple_value=self._stats_act_noise_mean())
    summary.value.add(tag="train/act_noise_std",  simple_value=self._stats_act_noise_std())


  def _reset(self):
    self.action_noise.reset()


  def _stats_act_noise_mean(self, *_):
    if len(self.act_noise_stats) == 0:
      return float("nan")
    return np.mean(self.act_noise_stats)

  def _stats_act_noise_std(self, *_):
    if len(self.act_noise_stats) == 0:
      return float("nan")
    return np.std(self.act_noise_stats)


  def _get_feed_dict(self, batch, t):
    feed_dict = {
      self.model.obs_t_ph:              batch["obs"],
      self.model.act_t_ph:              batch["act"],
      self.model.rew_t_ph:              batch["rew"],
      self.model.obs_tp1_ph:            batch["obs_tp1"],
      self.model.done_ph:               batch["done"],
      self.model.actor_opt_conf.lr_ph:  self.model.actor_opt_conf.lr_value(t),
      self.model.critic_opt_conf.lr_ph: self.model.critic_opt_conf.lr_value(t),
    }
    return feed_dict


  def _action_train(self, state, t):
    noise   = self.action_noise.sample(t)
    data    = self.model.action_train_ops(self.sess, state)
    action  = data["action"][0]
    action  = action + noise

    # Add action noise to stats
    self.act_noise_stats.append(noise)

    return action


  def _action_eval(self, state):
    data    = self.model.action_eval_ops(self.sess, state)
    action  = data["action"][0]
    return action


  def _state_action_spec(self, stack_frames):
    assert isinstance(self.env_train.observation_space, gym.spaces.Box)
    assert isinstance(self.env_train.action_space,      gym.spaces.Box)

    # Get environment specs
    act_shape = list(self.env_train.action_space.shape)
    obs_shape = list(self.env_train.observation_space.shape)

    if len(obs_shape) == 3:
      assert stack_frames > 1
      obs_dtype = np.uint8
      obs_len   = stack_frames
    else:
      obs_dtype = np.float32
      obs_len   = 1

    return obs_shape, obs_dtype, obs_len, act_shape
