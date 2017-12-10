import collections

import gym
import numpy as np
import tensorflow as tf

from rltf.agents.agent  import OffPolicyAgent
from rltf.memory        import ReplayBuffer


class AgentDDPG(OffPolicyAgent):

  def __init__(self,
               model_type,
               model_kwargs,
               actor_opt_conf,
               critic_opt_conf,
               action_noise,
               update_target_freq=1,
               memory_size=int(1e6),
               obs_hist_len=1,
               **agent_kwargs
              ):
    """
    Args:
      agent_config: dict. Dictionary with parameters for the Agent class. Must
        contain all parameters that do not have default values
      model_type: rltf.models.Model. TF implementation of a model network
      model_kwargs: dict. Model-specific keyword arguments to pass to the model
      actor_opt_conf: rltf.optimizers.OptimizerConf. Config for the actor network optimizer
      critic_opt_conf: rltf.optimizers.OptimizerConf. Config for the critic network optimizer
      action_noise: rltf.exploration.ExplorationNoise. Action exploration noise
        to add to the selected action
      memory_size: int. Size of the replay buffer
      obs_hist_len: int. How many environment observations comprise a single state.
    """

    super().__init__(**agent_kwargs)

    assert isinstance(self.env.observation_space, gym.spaces.Box)
    assert isinstance(self.env.action_space,      gym.spaces.Box)
    assert update_target_freq % self.train_freq == 0

    self.action_noise = action_noise

    self.actor_opt_conf   = actor_opt_conf
    self.critic_opt_conf  = critic_opt_conf
    self.update_target_freq = update_target_freq

    # Get environment specs
    act_shape = list(self.env.action_space.shape)
    obs_shape = list(self.env.observation_space.shape)

    # Image observation
    if len(obs_shape) == 3:
      obs_dtype = np.uint8
      obs_shape[-1] *= obs_hist_len
    else:
      obs_dtype = np.float32

    model_kwargs["obs_shape"]       = obs_shape
    model_kwargs["n_actions"]       = act_shape[0]
    model_kwargs["actor_opt_conf"]  = actor_opt_conf
    model_kwargs["critic_opt_conf"] = critic_opt_conf

    self.model      = model_type(**model_kwargs)
    self.replay_buf = ReplayBuffer(memory_size, obs_shape, obs_dtype, act_shape, np.float32, obs_hist_len)

    # Configure what information to log
    super()._build_log_info()

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
    tf.summary.scalar("actor_learn_rate",  self.actor_learn_rate_ph)
    tf.summary.scalar("critic_learn_rate", self.critic_learn_rate_ph)


  def _restore(self, graph):
    self.actor_learn_rate_ph  = graph.get_tensor_by_name("actor_learn_rate_ph:0")
    self.critic_learn_rate_ph = graph.get_tensor_by_name("critic_learn_rate_ph:0")


  def _custom_log_info(self):
    t = self.log_freq
    log_info = [
      ( "train/actor_learn_rate",           "f", self.actor_opt_conf.lr_value   ),
      ( "train/critic_learn_rate",          "f", self.critic_opt_conf.lr_value  ),
      ( "mean/act_noise_mean (%d steps)"%t, "f", self._stats_act_noise_mean     ),
      ( "mean/act_noise_std  (%d steps)"%t, "f", self._stats_act_noise_std      ),
    ]
    return log_info


  def reset(self):
    self.action_noise.reset()


  def _stats_act_noise_mean(self, *args):
    if len(self.act_noise_stats) == 0:
      return float("nan")
    return np.mean(self.act_noise_stats)

  def _stats_act_noise_std(self, *args):
    if len(self.act_noise_stats) == 0:
      return float("nan")
    return np.std(self.act_noise_stats)


  def _run_env(self):

    last_obs  = self.env.reset()

    for t in range (self.start_step, self.max_steps+1):
      # sess.run(t_inc_op)

      # Wait until net_thread is done
      self._wait_train_done()

      # Store the latest obesrvation in the buffer
      idx = self.replay_buf.store_frame(last_obs)

      # Get an action to run
      if self.learn_started:
        noise   = self.action_noise.sample()
        state   = self.replay_buf.encode_recent_obs()
        action  = self.model.control_action(self.sess, state)
        action  = action + noise
  
        # Add action noise to stats
        self.act_noise_stats.append(noise)

      else:
        # Choose random action when model not initialized
        action = self.env.action_space.sample()

      # Signal to net_thread that action is chosen
      self._signal_act_chosen()

      # Increement the TF timestep variable
      self.sess.run(self.t_tf_inc)

      # Run action
      # next_obs, reward, done, info = self.env.step(action)
      last_obs, reward, done, _ = self.env.step(action)

      # Store the effect of the action taken upon last_obs
      # self.replay_buf.store(obs, action, reward, done)
      self.replay_buf.store_effect(idx, action, reward, done)

      # Reset the environment if end of episode
      # if done: next_obs = self.env.reset()
      # obs = next_obs
      if done:
        last_obs = self.env.reset()
        self.reset()

      self._log_progress(t)


  def _train_model(self):

    for t in range (self.start_step, self.max_steps+1):

      if (t >= self.start_train and t % self.train_freq == 0):

        self.learn_started = True

        # Sample the Replay Buffer
        batch = self.replay_buf.sample(self.batch_size)

        # Compose feed_dict
        feed_dict = {
          self.model.obs_t_ph:       batch["obs"],
          self.model.act_t_ph:       batch["act"],
          self.model.rew_t_ph:       batch["rew"],
          self.model.obs_tp1_ph:     batch["obs_tp1"],
          self.model.done_ph:        batch["done"],
          self.actor_learn_rate_ph:  self.actor_opt_conf.lr_value(t),
          self.critic_learn_rate_ph: self.critic_opt_conf.lr_value(t),
          self.mean_ep_rew_ph:       self.mean_ep_rew,
          self.best_mean_ep_rew_ph:  self.best_mean_ep_rew,
        }

        self._wait_act_chosen()

        # Run a training step
        self.summary, _ = self.sess.run([self.summary_op, self.model.train_op], feed_dict=feed_dict)

        # Update target network
        if t % self.update_target_freq == 0:
          self.sess.run(self.model.update_target)

      else:
        self._wait_act_chosen()

      if t % self.save_freq == 0: self._save()

      self._signal_train_done()
