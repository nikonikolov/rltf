import collections

import gym
import numpy as np
import tensorflow as tf

from rltf.agents.off_pi import ParallelOffPolicyAgent
from rltf.agents.off_pi import SequentialOffPolicyAgent
from rltf.memory        import ReplayBuffer


class AgentDDPG(ParallelOffPolicyAgent):

  def __init__(self,
               model,
               model_kwargs,
               actor_opt_conf,
               critic_opt_conf,
               action_noise,
               update_target_freq=1,
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
      actor_opt_conf: rltf.optimizers.OptimizerConf. Config for the actor network optimizer
      critic_opt_conf: rltf.optimizers.OptimizerConf. Config for the critic network optimizer
      action_noise: rltf.exploration.ExplorationNoise. Action exploration noise
        to add to the selected action
      update_target_freq: Period in number of parameter updates (not steps!) at which to update the
        target net. Results in the period `train_freq * update_target_freq` number in number of step
      memory_size: int. Size of the replay buffer
      obs_len: int. How many environment observations comprise a single state.
    """

    super().__init__(**agent_kwargs)

    assert isinstance(self.env_train.observation_space, gym.spaces.Box)
    assert isinstance(self.env_train.action_space,      gym.spaces.Box)

    self.action_noise = action_noise

    self.actor_opt_conf   = actor_opt_conf
    self.critic_opt_conf  = critic_opt_conf
    self.update_target_freq = update_target_freq * self.train_freq

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
    model_kwargs["actor_opt_conf"]  = actor_opt_conf
    model_kwargs["critic_opt_conf"] = critic_opt_conf

    self.model      = model(**model_kwargs)
    self.replay_buf = ReplayBuffer(memory_size, obs_shape, obs_dtype, act_shape, np.float32, obs_len)

    # Configure what information to log
    self._define_log_info()

    # Custom TF Tensors and Ops
    self.actor_learn_rate_ph  = None
    self.critic_learn_rate_ph = None

    # Custom stats
    self.act_noise_stats = collections.deque([], maxlen=self.log_freq)


  def _build(self):
    # Create Learning rate placeholders
    self.actor_learn_rate_ph  = tf.placeholder(tf.float32, shape=(), name="actor_learn_rate_ph")
    self.critic_learn_rate_ph = tf.placeholder(tf.float32, shape=(), name="critic_learn_rate_ph")

    # Set the learn rate placeholders for the model
    self.actor_opt_conf.lr_ph  = self.actor_learn_rate_ph
    self.critic_opt_conf.lr_ph = self.critic_learn_rate_ph

    # Create learn rate summaries
    tf.summary.scalar("train/actor_learn_rate",  self.actor_learn_rate_ph)
    tf.summary.scalar("train/critic_learn_rate", self.critic_learn_rate_ph)


  def _restore(self, graph):
    self.actor_learn_rate_ph  = graph.get_tensor_by_name("actor_learn_rate_ph:0")
    self.critic_learn_rate_ph = graph.get_tensor_by_name("critic_learn_rate_ph:0")
    super()._restore(graph)


  def _append_log_info(self):
    t = self.log_freq
    log_info = [
      ( "train/actor_learn_rate",           "f", self.actor_opt_conf.lr_value   ),
      ( "train/critic_learn_rate",          "f", self.critic_opt_conf.lr_value  ),
      ( "mean/act_noise_mean (%d steps)"%t, "f", self._stats_act_noise_mean     ),
      ( "mean/act_noise_std  (%d steps)"%t, "f", self._stats_act_noise_std      ),
    ]
    return log_info


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
      self.model.obs_t_ph:       batch["obs"],
      self.model.act_t_ph:       batch["act"],
      self.model.rew_t_ph:       batch["rew"],
      self.model.obs_tp1_ph:     batch["obs_tp1"],
      self.model.done_ph:        batch["done"],
      self.actor_learn_rate_ph:  self.actor_opt_conf.lr_value(t),
      self.critic_learn_rate_ph: self.critic_opt_conf.lr_value(t),
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
