import logging
import os
import time
import threading

import numpy      as np
import tensorflow as tf

import rltf.conf
import rltf.log
from rltf.env_wrap.utils import get_monitor_wrapper

stats_logger  = logging.getLogger(rltf.conf.STATS_LOGGER_NAME)
logger        = logging.getLogger(__name__)


class Agent:
  """The base class for a Reinforcement Learning agent"""

  def __init__(self,
               env,
               train_freq,
               start_train,
               max_steps,
               batch_size,
               model_dir,
               log_freq=10000,
               save=False,
               save_freq=int(1e5),
              ):
    """
    Args:
      env: gym.Env. Environment in which the model will be trained.
      model_dir: string. Directory path for the model logs and checkpoints
      start_train: int. Time step at which the agent starts learning
      max_steps: int. Training step at which learning stops
      train_freq: int. How many environment actions to take between every 2 learning steps
      batch_size: int. Batch size for training the model
      log_freq: int. Add TensorBoard summary and print progress every log_freq
        number of environment steps
      save: bool. If true, save the model every
      save_freq: int. Save the model every `save_freq` training steps


      exploration: rltf.schedules.Schedule. Exploration schedule for the model
    """

    # Store parameters
    self.env          = env
    self.env_monitor  = get_monitor_wrapper(env)

    self.train_freq   = train_freq
    self.start_train  = start_train
    self.max_steps    = max_steps

    self.batch_size   = batch_size
    self.model_dir    = model_dir

    self.log_freq     = log_freq
    self.save         = save
    self.save_freq    = save_freq


    self.env_file     = os.path.join(self.model_dir, "Env.pkl")

    # Stats
    self.episode_rewards  = np.asarray([])
    self.mean_ep_rew      = -float('nan')
    self.best_mean_ep_rew = -float('inf')
    self.stats_n          = 100   # Number of episodes over which to take stats
    self.summary          = None

    # Attributes that are set during build
    self.model            = None
    self.start_step       = None
    self.learn_started    = None
    self.log_info         = None
    self.last_log_time    = None

    self.t_tf             = None
    self.t_tf_inc         = None
    self.summary_op       = None
    self.mean_ep_rew_ph   = None
    self.best_mean_ep_rew_ph = None

    self.sess             = None
    self.saver            = None
    self.tb_writer        = None


  def build(self):
    """Build the graph. If there is already a checkpoint in `self.model_dir`,
    then it will be restored instead. Calls either `self._build()` and
    `self.model.build()` or `self._restore()` and `self.model.restore()`.
    """

    # Check for checkpoint
    ckpt    = tf.train.get_checkpoint_state(self.model_dir)
    restore = ckpt and ckpt.model_checkpoint_path

    # ------------------ BUILD THE MODEL ----------------
    if not restore:
      logger.info("Building model")
      # tf.reset_default_graph()

      # Call the subclass _build function
      self._build()

      # Build the model
      self.model.build()

      # Create timestep variable and logs placeholders
      with tf.device('/cpu:0'):
        self.mean_ep_rew_ph      = tf.placeholder(tf.float32, shape=(), name="mean_ep_rew_ph")
        self.best_mean_ep_rew_ph = tf.placeholder(tf.float32, shape=(), name="best_mean_ep_rew_ph")

        self.t_tf                = tf.Variable(1, dtype=tf.int32, trainable=False, name="t_tf")
        self.t_tf_inc            = tf.assign(self.t_tf, self.t_tf + 1, name="t_inc_op")

        tf.summary.scalar("mean_ep_rew",      self.mean_ep_rew_ph)
        tf.summary.scalar("best_mean_ep_rew", self.best_mean_ep_rew_ph)

        # Create an Op for all summaries
        self.summary_op = tf.summary.merge_all()

      # Set control variables
      self.start_step     = 1
      self.learn_started  = False

      # Create a session and initialize the model
      self.sess = self._get_sess()
      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.local_variables_initializer())

      # Initialize the model
      self.model.initialize(self.sess)

    # ------------------ RESTORE THE MODEL ----------------
    else:
      logger.info("Restoring model")

      # Restore the graph
      ckpt_path = ckpt.model_checkpoint_path + '.meta'
      saver = tf.train.import_meta_graph(ckpt_path)
      graph = tf.get_default_graph()

      # Get the general variables and placeholders
      self.t_tf                 = graph.get_tensor_by_name("t_tf:0")
      self.t_tf_inc             = graph.get_operation_by_name("t_tf_inc")
      self.mean_ep_rew_ph       = graph.get_tensor_by_name("mean_ep_rew_ph:0")
      self.best_mean_ep_rew_ph  = graph.get_tensor_by_name("best_mean_ep_rew_ph:0")

      # Restore the model variables
      self.model.restore(graph)

      # Restore the agent subclass variables
      self._restore(graph)

      # Restore the session
      self.sess = self._get_sess()
      saver.restore(self.sess, ckpt.model_checkpoint_path)

      # Get the summary Op
      self.summary_op = graph.get_tensor_by_name("Merge/MergeSummary:0")

      # Set control variables
      self.start_step     = self.sess.run(self.t_tf)
      self.learn_started  = True

    # Create the Saver object: NOTE that you must do it after building the whole graph
    self.saver     = tf.train.Saver(max_to_keep=2, save_relative_paths=True)
    self.tb_writer = tf.summary.FileWriter(self.model_dir + "tb/", self.sess.graph)


  def train(self):
    raise NotImplementedError()


  def reset(self):
    raise NotImplementedError()

  def close(self):
    # Close session on exit
    self.tb_writer.close()
    self.sess.close()


  def _build(self):
    """Used by the subclass to build class specific TF objects. Must not call
    `self.model.build()`
    """
    raise NotImplementedError()

  def _get_feed_dict(self):
    """Get the placeholder parameters to feed to the model while training"""
    raise NotImplementedError()


  def _restore(self, graph):
    """Restore the Variables, placeholders and Ops needed by the class so that
    it can operate in exactly the same way as if `self._build()` was called

    Args:
      graph: tf.Graph. Graph, restored from a checkpoint
    """
    raise NotImplementedError()


  def _build_log_info(self):
    n = self.stats_n

    default_info = [
      ("total/agent_steps",                     "d",    lambda t: t),
      ("total/env_steps",                       "d",    lambda t: self.env_monitor.get_total_steps()),
      ("total/episodes",                        "d",    lambda t: self.episode_rewards.size),

      ("mean/n_eps > 0.8*best_rew (%d eps)"%n,  ".3f",  self._stats_frac_good_episodes),
      ("mean/ep_length",                        ".3f",  self._stats_ep_length),
      ("mean/steps_per_sec",                    ".3f",  self._stats_steps_per_sec),
      ("mean/reward (%d eps)"%n,                ".3f",  lambda t: self.mean_ep_rew),

      ("best/episode_reward",                   ".3f",  self._stats_best_reward),
      ("best/mean_reward (%d eps)"%n,           ".3f",  lambda t: self.best_mean_ep_rew),
    ]

    custom_log_info = self._custom_log_info()
    log_info = default_info + custom_log_info

    self.log_info = rltf.log.format_tabular(log_info)


  def _stats_ep_length(self, *args):
    ep_lengths = self.env_monitor.get_episode_lengths()
    if len(ep_lengths) > 0:
      return np.mean(ep_lengths)
    return float("nan")

  def _stats_frac_good_episodes(self, *args):
    if self.episode_rewards.size == 0:
      return float("nan")
    ep_rews   = self.episode_rewards[-self.stats_n:]
    best_rew  = ep_rews.max()
    if best_rew >= 0:
      thresh  = 0.8 * best_rew
    else:
      thresh  = 1.2 * best_rew
    good_eps  = ep_rews >= thresh
    return np.sum(good_eps) / float(self.stats_n)

  def _stats_best_reward(self, *args):
    if self.episode_rewards.size == 0:
      return float("nan")
    return self.episode_rewards.max()

  def _stats_steps_per_sec(self, *args):
    now  = time.time()
    if self.last_log_time is None:
      t_per_s = float("nan")
    else:
      t_per_s = self.log_freq / (now - self.last_log_time)
    self.last_log_time  = now
    return t_per_s


  def _custom_log_info(self):
    """
    Returns:
      List of tuples `(name, format, lambda)` with information of custom subclass
      parameters to log during training. `name`: `str`, the name of the reported
      value. `modifier`: `str`, the type modifier for printing the value.
      `lambda`: A function that takes the current timestep as argument and
      returns the value to be printed.
    """
    raise NotImplementedError()


  def _log_progress(self, t):
    """Log the training progress and append the TensorBoard summary.
    Note that the TensorBoard summary might be 1 step older.

    Args:
      t: int. Current timestep
    """

    # Run the update only 2 step before the actual logging happens in order to
    # make sure that the most recent possible values will be stored in
    # self.summary. This is a hacky workaround in order to support OffPolicyAgent
    # which runs 2 threads without coordination
    if (t+2) % self.log_freq == 0 and self.learn_started:
      episode_rewards = self.env_monitor.get_episode_rewards()
      self.episode_rewards = np.asarray(episode_rewards)
      if self.episode_rewards.size > 0:
        self.mean_ep_rew      = np.mean(episode_rewards[-self.stats_n:])
        self.best_mean_ep_rew = max(self.best_mean_ep_rew, self.mean_ep_rew)

    if t % self.log_freq == 0 and self.learn_started:
      stats_logger.info("")
      for s, lambda_v in self.log_info:
        stats_logger.info(s.format(lambda_v(t)))
      stats_logger.info("")

      if self.summary:
        # Log with TensorBoard
        self.tb_writer.add_summary(self.summary, global_step=t)


  def _save(self):
    # Save model
    if self.learn_started and self.save:
      logger.info("Saving model")
      self.saver.save(self.sess, self.model_dir, global_step=self.t_tf)
      # logger.info("Saving memory")
      # self.replay_buf.save(self.model_dir)
      # pickle_save(self.env_file, self.env)


  def _get_sess(self):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)



class OffPolicyAgent(Agent):
  """The base class for Off-policy agents

  Allows to run env actions and train the model in separate threads, while
  providing an easy way to synchronize between the threads. Can speed up
  training by 20-50%
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Create synchronization events
    self._act_chosen       = threading.Event()
    self._train_done       = threading.Event()


  def train(self):

    self._act_chosen.clear()
    self._train_done.set()

    env_thread  = threading.Thread(name='environment_thread', target=self._run_env)
    nn_thread   = threading.Thread(name='network_thread',     target=self._train_model)

    nn_thread.start()
    env_thread.start()

    # Wait for threads
    env_thread.join()
    nn_thread.join()

    # self.tb_writer.close()
    # self.sess.close()


  def _action_train(self):
    """Return action selected by the agent for a training step"""
    raise NotImplementedError()


  def _action_eval(self):
    """Return action selected by the agent for an evaluation step"""
    raise NotImplementedError()


  def _run_env(self):
    """Thread for running the environment. Must call `self._wait_train_done()`
    before selcting an action (by running the model). This ensures that the
    `self._train_model()` thread has finished the training step. After action
    is selected, it must call `self._signal_act_chosen()` to allow
    `self._train_model()` thread to start a new training step
    """

    last_obs  = self.env.reset()

    for t in range (self.start_step, self.max_steps+1):
      # sess.run(t_inc_op)

      # Wait until net_thread is done
      self._wait_train_done()

      # Store the latest obesrvation in the buffer
      idx = self.replay_buf.store_frame(last_obs)

      # Get an action to run
      if self.learn_started:
        action = self._action_train()

      # Choose random action if learning has not started
      else:
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
    """Thread for trianing the model. Must call `self._wait_act_chosen()`
    before trying to run a training step on the model. This ensures that the
    `self._run_env()` thread has finished selcting an action (by running the model).
    After training step is done, it must call `self._signal_train_done()` to allow
    `self._run_env()` thread to select a new action
    """

    for t in range (self.start_step, self.max_steps+1):

      if (t >= self.start_train and t % self.train_freq == 0):

        self.learn_started = True

        # Compose feed_dict
        feed_dict = self._get_feed_dict()

        self._wait_act_chosen()

        # Run a training step
        self.summary, _ = self.sess.run([self.summary_op, self.model.train_op], feed_dict=feed_dict)

        # Update target network
        if t % self.update_target_freq == 0:
          self.sess.run(self.model.update_target)

      else:
        self._wait_act_chosen()

      if t % self.save_freq == 0:
        self._save()

      self._signal_train_done()


  def _wait_act_chosen(self):
    # Wait until an action is chosen to be run
    while not self._act_chosen.is_set():
      self._act_chosen.wait()
    self._act_chosen.clear()

  def _wait_train_done(self):
    # Wait until training step is done
    while not self._train_done.is_set():
      self._train_done.wait()
    self._train_done.clear()

  def _signal_act_chosen(self):
    # Signal that the action is chosen and the TF graph is safe to be run
    self._act_chosen.set()

  def _signal_train_done(self):
    # Signal to main that the training step is done running
    self._train_done.set()
