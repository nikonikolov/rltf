import gym
import numpy as np

from rltf.agents      import LoggingAgent
from rltf.memory      import PGBuffer
from rltf.monitoring  import Monitor


class AgentPG(LoggingAgent):

  def __init__(self,
               env_maker,
               model,
               gamma,
               lam,
               epochs,
               vf_iters=1,
               stack_frames=3,
               **agent_kwargs
              ):
    """
    Args:
      env_maker: callable. Function that takes the mode of an env and retruns a new environment instance
      gamma: float. Discount factor for GAE(gamma, lambda)
      lam: float. Lambda value for GAE(gamma, lambda)
      epochs: int. Number of training epochs. An epoch comprises collecting experience with the current
        policy, running several training steps to update the policy and throwing the experience away
      vf_iters: int. Number of value function training steps in a single epoch
    """

    super().__init__(**agent_kwargs)

    self.env_train = Monitor(
                      env=env_maker('t'),
                      log_dir=self.model_dir,
                      mode='t',
                      log_period=None,
                      video_spec=self.video_period,
                      epochs=True,
                    )

    self.env_eval  = Monitor(
                      env=env_maker('e'),
                      log_dir=self.model_dir,
                      mode='e',
                      log_period=self.eval_len,
                      video_spec=self.video_period,
                      eval_period=self.eval_period,
                      epochs=True,
                    )


    self.gamma    = gamma
    self.lam      = lam
    self.vf_iters = vf_iters
    self.epochs   = epochs

    # Get environment specs
    obs_shape, obs_dtype, act_shape, act_dtype, obs_len = self._state_action_spec(stack_frames)

    # Initialize the model and the experience buffer
    self.model  = model(obs_shape=obs_shape, act_space=self.env_train.action_space, **self.model_kwargs)
    self.buffer = PGBuffer(self.batch_size, obs_shape, obs_dtype, act_shape, act_dtype, obs_len)


  def _train(self):
    # Get the function that generates trajectories
    run_policy = self._trajectory_generator(self.batch_size)

    for t in range(self.epochs):
      if self._terminate:
        break

      # Collect experience in the environment
      run_policy()

      # Train the model
      self._run_train_step(t)

      self.env_train.monitor.log_stats()

      # Stop and run evaluation procedure
      if self.eval_len > 0 and t % self.eval_period == 0:
        self._eval_agent()

      # Update the agent step - corresponds to number of epochs
      self.agent_step = t

      # Save **after** agent step is correct and completed
      if t % self.save_period == 0:
        self.save()


  def _trajectory_generator(self, horizon):
    """
    Args:
      horizon: int. Number of steps to run before yielding the trajectories
    Returns:
      A function which generates trajectories
    """

    obs = self.reset()

    def run_env():
      nonlocal obs

      # Clear the buffer to avoid using old data
      self.buffer.reset()

      for _ in range(horizon):
        if self._terminate:
          return

        # Get an action to run and the value function estimate of this state
        action, vf, logp = self._action_train(obs)

        # Run action
        next_obs, reward, done, info = self.env_train.step(action)

        # Store the effect of the action taken upon obs
        self.buffer.store(obs, action, reward, done, vf, logp)

        # Reset the environment if end of episode
        if done:
          next_obs = self.reset()
        obs = next_obs

      # Store the value function for the next state. Needed to compute GAE(lambda)
      if not done:
        _, next_vf, _ = self._action_train(obs)
      else:
        next_vf = 0

      # Compute GAE(gamma, lambda) and TD(lambda)
      self.buffer.compute_estimates(self.gamma, self.lam, next_vf)

    return run_env


  def _get_feed_dict(self, batch, t):
    feed_dict = {
      self.model.obs_ph:        batch["obs"],
      self.model.act_ph:        batch["act"],
      self.model.adv_ph:        batch["adv"],
      self.model.ret_ph:        batch["ret"],
      self.model.old_logp_ph:   batch["logp"],
      self.model.pi_opt_conf.lr_ph:  self.model.pi_opt_conf.lr_value(t),
      self.model.vf_opt_conf.lr_ph:  self.model.vf_opt_conf.lr_value(t),
    }

    return feed_dict


  def _run_summary_op(self, t, feed_dict):
    # Run summary after every training epoch
    self.summary = self.sess.run(self.summary_op, feed_dict=feed_dict)


  def _run_train_step(self, t):
    batch     = self.buffer.get_data()
    feed_dict = self._get_feed_dict(batch, t)

    train_pi  = self.model.ops_dict["train_pi"]
    train_vf  = self.model.ops_dict["train_vf"]

    # Run a policy gradient step and a value function training step
    self.sess.run([train_pi, train_vf], feed_dict=feed_dict)
    # self.sess.run([self.model.train_op], feed_dict=feed_dict)

    # Run a policy gradient step
    # self.sess.run(train_pi, feed_dict=feed_dict)

    # Train the value function additionally
    for _ in range(self.vf_iters-1):
      if self._terminate:
        break
      self.sess.run(train_vf, feed_dict=feed_dict)

    # Run the summary op to log the changes from the update if necessary
    self._run_summary_op(t, feed_dict)


  def _action_train(self, state):
    data   = self.model.action_train_ops(self.sess, state)
    action = data["action"][0]
    vf     = data["vf"][0]
    logp   = data["logp"][0]
    return action, vf, logp


  def _action_eval(self, state):
    data   = self.model.action_eval_ops(self.sess, state)
    action = data["action"][0]
    return action


  def _save_allowed(self):
    # Prevent saving if the process was terminated - state is most likely inconsistent
    return not self._terminate


  def _state_action_spec(self, stack_frames):
    assert isinstance(self.env_train.observation_space, gym.spaces.Box)

    # Get environment specs
    act_shape = list(self.env_train.action_space.shape)
    obs_shape = list(self.env_train.observation_space.shape)

    # Get obs_shape and obs_dtype
    if len(obs_shape) == 3:
      assert stack_frames > 1
      obs_dtype = np.uint8
      obs_len   = stack_frames
    else:
      obs_dtype = np.float32
      obs_len   = 1

    # Get act_shape and act_dtype
    if isinstance(self.env_train.action_space, gym.spaces.Box):
      act_shape = list(self.env_train.action_space.shape)
      act_dtype = np.float32
    elif isinstance(self.env_train.action_space, gym.spaces.Discrete):
      act_shape = []
      act_dtype = np.uint8
    else:
      raise ValueError("Unsupported action space")

    return obs_shape, obs_dtype, act_shape, act_dtype, obs_len


  def _reset(self):
    pass
