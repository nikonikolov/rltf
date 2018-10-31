import logging
import gym
import numpy as np
import tensorflow as tf

from rltf.agents  import ParallelOffPolicyAgent
from rltf.agents  import SequentialOffPolicyAgent
from rltf.memory  import ReplayBuffer


logger = logging.getLogger(__name__)


class AgentDQN(ParallelOffPolicyAgent):

  def __init__(self,
               model,
               model_kwargs,
               exploration,
               update_target_freq=10000,
               memory_size=int(1e6),
               obs_len=4,
               epsilon_eval=0.001,
               **agent_kwargs
              ):
    """
    Args:
      model: rltf.models.Model. TF implementation of a model network
      model_kwargs: dict. Model-specific keyword arguments to pass to the model
      exploration: rltf.schedules.Schedule. Epsilon value for e-greedy exploration
      update_target_freq: Period in number of agent steps at which to update the target net
      memory_size: int. Size of the replay buffer
      obs_len: int. How many environment observations comprise a single state.
      agent_kwargs: Keyword arguments that will be passed to the Agent base class
    """

    super().__init__(**agent_kwargs)

    assert isinstance(self.env_train.observation_space, gym.spaces.Box)
    assert isinstance(self.env_train.action_space,      gym.spaces.Discrete)

    self.exploration = exploration
    self.epsilon_eval = epsilon_eval
    self.update_target_freq = update_target_freq

    # Get environment specs
    n_actions = self.env_train.action_space.n
    obs_shape = self.env_train.observation_space.shape
    obs_shape = list(obs_shape)
    # obs_len   = obs_len if len(obs_shape) == 3 else 1
    if len(obs_shape) != 3:
      obs_len = 1
      logger.warning("Overriding obs_len value since env has low-level observations space ")

    model_kwargs["obs_shape"] = obs_shape
    model_kwargs["n_actions"] = n_actions

    self.model      = model(**model_kwargs)
    self.replay_buf = ReplayBuffer(memory_size, obs_shape, np.uint8, [], np.uint8, obs_len)


  def _build(self):
    pass


  def _append_log_spec(self):
    return []


  def _append_summary(self, summary, t):
    summary.value.add(tag="train/epsilon", simple_value=self.exploration.value(t))


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
    epsilon = self.exploration.value(t)
    if self.prng.uniform(0,1) < epsilon:
      action = self.env_train.action_space.sample()
    else:
      # Run the network to select an action
      action = self.model.action_train(self.sess, state)
    return action


  def _action_eval(self, state, t):
    # Run epsilon greedy policy
    if self.prng.uniform(0,1) < self.epsilon_eval:
      action = self.env_eval.action_space.sample()
    else:
      # Run the network to select an action
      action = self.model.action_eval(self.sess, state)
    return action


  def _reset(self):
    pass
