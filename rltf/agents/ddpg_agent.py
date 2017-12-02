import gym
import numpy as np
import tensorflow as tf

from rltf.agents.agent  import OffPolicyAgent
from rltf.memory        import ReplayBuffer


class AgentDDPG(OffPolicyAgent):

  def __init__(self,
               agent_config,
               model_type,
               model_kwargs,
               actor_opt_conf,
               critic_opt_conf,
               action_noise,
               memory_size=int(1e6),
               obs_hist_len=1,
              ):
    """
    Args:
      agent_config: dict. Dictionary with parameters for the Agent class. Must
        contain all parameters that do not have default values
      model_type: rltf.models.Model. TF implementation of a model network
      model_kwargs: dict. Model-specific keyword arguments to pass to the model
      actor_opt_conf: rltf.optimizers.OptimizerConf. Config for the actor network optimizer
      critic_opt_conf: rltf.optimizers.OptimizerConf. Config for the critic network optimizer
      action_noise: rltf.exploration.ExplorationNoise. Ation exploration noise 
        to add to the selected action
      memory_size: int. Size of the replay buffer
      obs_hist_len: int. How many environment observations comprise a single state.
    """
  
    super().__init__(**agent_config)

    assert type(self.env.observation_space) == gym.spaces.Box
    assert type(self.env.action_space)      == gym.spaces.Box

    # Get environment specs
    act_shape = list(self.env.action_space.shape)
    obs_shape = list(self.env.observation_space.shape)
    
    # Image observation
    if len(obs_shape) == 3:
      obs_dtype = np.uint8
      obs_shape[-1] *= obs_hist_len
    else:
      obs_dtype = np.float32

    self.act_min   = self.env.action_space.low
    self.act_max   = self.env.action_space.high
    self.action_noise = action_noise

    self.actor_opt_conf   = actor_opt_conf
    self.critic_opt_conf  = critic_opt_conf

    # self.actor_lr  = actor_opt_conf.lr_schedule
    # self.critic_lr = critic_opt_conf.lr_schedule

    model_kwargs["obs_shape"]       = obs_shape
    model_kwargs["act_min"]         = self.act_min
    model_kwargs["act_max"]         = self.act_max
    model_kwargs["actor_opt_conf"]  = actor_opt_conf
    model_kwargs["critic_opt_conf"] = critic_opt_conf
 
    # model_kwargs = dict(obs_shape=obs_shape,
    #                     n_actions=n_actions,
    #                     act_min=self.act_min,
    #                     act_max=self.act_max,
    #                     actor_opt_conf=actor_opt_conf,
    #                     critic_opt_conf=critic_opt_conf,
    #                     tau=tau,
    #                     gamma=gamma,
    #                     huber_loss=huber_loss
    #                    )

    self.model      = model_type(**model_kwargs)
    self.replay_buf = ReplayBuffer(memory_size, obs_shape, obs_dtype, act_shape, np.float32, obs_hist_len)
    
    # Configure what information to log
    self._build_log_list()


  def build(self):
    # Create Learning rate placeholders
    self.actor_learn_rate_ph  = tf.placeholder(tf.float32, shape=(), name="actor_learn_rate_ph")
    self.critic_learn_rate_ph = tf.placeholder(tf.float32, shape=(), name="critic_learn_rate_ph")

    # Set the placeholders for the model
    self.actor_opt_conf.lr_ph  = self.actor_learn_rate_ph
    self.critic_opt_conf.lr_ph = self.critic_learn_rate_ph

    # Create learn rate summaries
    tf.summary.scalar("actor_learn_rate",  self.actor_learn_rate_ph)
    tf.summary.scalar("critic_learn_rate", self.critic_learn_rate_ph)

    # Make the general configuration and get a tf.Session()
    super()._build()


  def _build_log_list(self):
    log_info = [
      ( "actor learn rate",  "%f", lambda t: self.actor_opt_conf.lr_value(t)  ),
      ( "critic learn rate", "%f", lambda t: self.critic_opt_conf.lr_value(t) ),
    ]    
    super()._build_log_list(log_info)


  def reset(self):
    self.action_noise.reset()


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

      else:
        # Choose random action when model not initialized
        action = self.env.action_space.sample()

      # Signal to net_thread that action is chosen
      self._signal_act_chosen()

      # Increement the TF timestep variable
      self.sess.run(self.t_tf_inc)

      # Run action
      # next_obs, reward, done, info = self.env.step(action)
      last_obs, reward, done, info = self.env.step(action)

      # Store the effect of the action taken upon last_obs
      # self.replay_buf.store(obs, action, reward, done)
      self.replay_buf.store_effect(idx, action, reward, done)

      # Reset the environment if end of episode
      # if done: next_obs = self.env.reset()
      if done: 
        last_obs = self.env.reset()
        self.reset()

      # obs = next_obs

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
        self.sess.run(self.model.update_target)

      else:
        self._wait_act_chosen()
  
      if t % self.save_freq == 0: self._save()

      self._signal_train_done()
