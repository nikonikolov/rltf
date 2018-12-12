import gym
import numpy as np

from rltf.agents      import QlearnAgent
from rltf.memory      import ReplayBuffer
from rltf.monitoring  import Monitor


class AgentDQN(QlearnAgent):

  def __init__(self,
               env_maker,
               model,
               epsilon_train,
               epsilon_eval,
               memory_size=int(1e6),
               stack_frames=4,
               **agent_kwargs
              ):
    """
    Args:
      env_maker: callable. Function that takes the mode of an env and retruns a new environment instance
      model: rltf.models.Model. TF implementation of a model network
      epsilon_train: rltf.schedules.Schedule. Epsilon value for e-greedy exploration
      epsilon_eval: float. Epsilon value for selecting random action during evaluation
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

    self.epsilon_train  = epsilon_train
    self.epsilon_eval = epsilon_eval

    # Get environment specs
    obs_shape, obs_dtype, obs_len, n_actions = self._state_action_spec(stack_frames)

    # Initialize the model and the experience buffer
    self.model      = model(obs_shape=obs_shape, n_actions=n_actions, **self.model_kwargs)
    self.replay_buf = ReplayBuffer(memory_size, obs_shape, obs_dtype, [], np.uint8, obs_len)


  def _append_summary(self, summary, t):
    summary.value.add(tag="train/epsilon", simple_value=self.epsilon_train.value(t))


  def _get_feed_dict(self, batch, t):
    feed_dict = {
      self.model.obs_t_ph:        batch["obs"],
      self.model.act_t_ph:        batch["act"],
      self.model.rew_t_ph:        batch["rew"],
      self.model.obs_tp1_ph:      batch["obs_tp1"],
      self.model.done_ph:         batch["done"],
      self.model.opt_conf.lr_ph:  self.model.opt_conf.lr_value(t),
    }
    return feed_dict


  def _action_train(self, state, t):
    # Run epsilon greedy policy
    epsilon = self.epsilon_train.value(t)
    if self.prng.uniform(0,1) < epsilon:
      action = self.env_train.action_space.sample()
    else:
      # Run the network to select an action
      data   = self.model.action_train_ops(self.sess, state)
      action = data["action"][0]
    return action


  def _action_eval(self, state):
    # Run epsilon greedy policy
    if self.prng.uniform(0,1) < self.epsilon_eval:
      action = self.env_eval.action_space.sample()
    else:
      # Run the network to select an action
      data   = self.model.action_eval_ops(self.sess, state)
      action = data["action"][0]
    return action


  def _reset(self):
    pass


  def _state_action_spec(self, stack_frames):
    assert isinstance(self.env_train.observation_space, gym.spaces.Box)
    assert isinstance(self.env_train.action_space,      gym.spaces.Discrete)

    n_actions = self.env_train.action_space.n
    obs_shape = self.env_train.observation_space.shape
    obs_shape = list(obs_shape)

    if len(obs_shape) == 3:
      assert stack_frames > 1
      obs_dtype = np.uint8
      obs_len   = stack_frames
    else:
      obs_dtype = np.float32
      obs_len   = 1

    return obs_shape, obs_dtype, obs_len, n_actions



class AgentBDQN(AgentDQN):

  def __init__(self, blr_train_period, blr_batch_size, **kwargs):
    """
    Args:
      blr_train_period: int. Period in number of steps at which to train the BLR
      blr_batch_size: int. Number of samples to train BLR in an update step
    """
    super().__init__(**kwargs)

    self.blr_train_period = blr_train_period
    self.blr_batch_size = blr_batch_size


  def _run_train_step(self, t):
    super()._run_train_step(t)

    # Train BLR
    if t % self.blr_train_period == 0:
      # for batch in self.replay_buf.random_data(self.blr_batch_size, self.batch_size):
      #   feed_dict = self._get_feed_dict(batch, t)
      #   self.sess.run(self.model.train_blr, feed_dict=feed_dict)

      n = int(self.blr_batch_size / (self.batch_size * 4))
      for _ in range(n):
        batch = self.replay_buf.sample(self.batch_size)
        feed_dict = self._get_feed_dict(batch, t)
        self.sess.run(self.model.train_blr, feed_dict=feed_dict)
