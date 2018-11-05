import collections

import gym
import numpy as np

from rltf.agents  import QlearnAgent
from rltf.memory  import ReplayBuffer


class AgentDDPG(QlearnAgent):

  def __init__(self,
               model,
               model_kwargs,
               action_noise,
               update_target_period=1,
               memory_size=int(1e6),
               obs_len=1,
               **agent_kwargs
              ):
    """
    Args:
      agent_config: dict. Dictionary with parameters for the Agent class. Must
        contain all parameters that do not have default values
      model: rltf.models.Model. TF implementation of a model network
      model_kwargs: dict. Model-specific keyword arguments to pass to the model
      action_noise: rltf.exploration.ExplorationNoise. Action exploration noise
        to add to the selected action
      update_target_period: Period in number of agent steps at which to update the target net
      memory_size: int. Size of the replay buffer
      obs_len: int. How many environment observations comprise a single state.
    """

    super().__init__(**agent_kwargs)

    assert isinstance(self.env_train.observation_space, gym.spaces.Box)
    assert isinstance(self.env_train.action_space,      gym.spaces.Box)

    self.action_noise = action_noise
    self.update_target_period = update_target_period

    # Get environment specs
    act_shape = list(self.env_train.action_space.shape)
    obs_shape = list(self.env_train.observation_space.shape)

    # Image observation
    if len(obs_shape) == 3:
      assert obs_len > 1
      obs_dtype = np.uint8
    else:
      obs_dtype = np.float32

    model_kwargs["obs_shape"]       = obs_shape
    model_kwargs["n_actions"]       = act_shape[0]

    self.model      = model(**model_kwargs)
    self.replay_buf = ReplayBuffer(memory_size, obs_shape, obs_dtype, act_shape, np.float32, obs_len)

    # Custom stats
    self.act_noise_stats = collections.deque([], maxlen=self.log_period)


  def _build(self):
    pass


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
    action  = self.model.action_train(self.sess, state)
    action  = action + noise

    # Add action noise to stats
    self.act_noise_stats.append(noise)

    return action


  def _action_eval(self, state, t):
    action  = self.model.action_eval(self.sess, state)
    return action
