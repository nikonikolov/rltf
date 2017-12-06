import os
import time
import threading
import sys

import numpy      as np
import tensorflow as tf

import rltf.env_wrappers.utils as env_utils


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
    self.env_monitor  = env_utils.get_monitor_wrapper(env)

    self.train_freq   = train_freq
    self.start_train  = start_train
    self.max_steps    = max_steps

    self.batch_size   = batch_size
    self.model_dir    = model_dir

    self.log_freq     = log_freq
    self.save         = save
    self.save_freq    = save_freq


    self.env_file     = os.path.join(self.model_dir, "Env.pkl")

    self.summary      = None

    self.episodes         = 0
    self.mean_ep_rew      = -float('nan')
    self.best_ep_rew      = -float('inf')
    self.best_mean_ep_rew = -float('inf')

    self.learn_started    = None

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
      print("Building model")
      # logger.info("Building model")
      tf.reset_default_graph()

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
      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.local_variables_initializer())

      # Initialize the model
      self.model.initialize(self.sess)

    # ------------------ RESTORE THE MODEL ----------------
    else:
      print("Restoring model")
      # logger.info("Restoring model")

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
      self.sess = tf.Session()
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


  def _restore(self, graph):
    """Restore the Variables, placeholders and Ops needed by the class so that
    it can operate in exactly the same way as if `self._build()` was called

    Args:
      graph: tf.Graph. Graph, restored from a checkpoint
    """
    raise NotImplementedError()


  def _build_log_info(self):

    def mean_step_time():
      if not hasattr(self, "last_log_time"):
        self.last_log_time = time.time()
        return float("nan")
      time_now  = time.time()
      mean_time = (time_now - self.last_log_time) / self.log_freq
      self.last_log_time  = time_now
      return mean_time

    default_info = [
      ("timestep",              "%d", lambda t: t),
      ("episodes",              "%d", lambda t: self.episodes),
      ("mean reward (100 eps)", "%f", lambda t: self.mean_ep_rew),
      ("best mean reward",      "%f", lambda t: self.best_mean_ep_rew),
      ("best episode reward",   "%f", lambda t: self.best_ep_rew),
      ("mean step time",        "%f", lambda t: mean_step_time()),
    ]

    custom_log_info = self._custom_log_info()
    log_info = default_info + custom_log_info

    str_sizes = [len(s) for s, _, _ in log_info]
    pad = max(str_sizes) + 2

    self.log_info  = [(s.ljust(pad) + ptype, v) for s, ptype, v in log_info]
    self.log_info  = [("=" * pad + "%s", lambda t: "")] + self.log_info


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
      if len(episode_rewards) > 0:
        self.mean_ep_rew      = np.mean(episode_rewards[-100:])
        self.best_ep_rew      = max(episode_rewards)
        self.best_mean_ep_rew = max(self.best_mean_ep_rew, self.mean_ep_rew)
      self.episodes = len(episode_rewards)

    if t % self.log_freq == 0 and self.learn_started:
      for s, lambda_v in self.log_info:
        print(s % lambda_v(t))
      sys.stdout.flush()

      if self.summary:
        # Log with TensorBoard
        self.tb_writer.add_summary(self.summary, global_step=t)


  def _save(self):
    # Save model
    if self.learn_started and self.save:
      print("Saving model")
      self.saver.save(self.sess, self.model_dir, global_step=self.t_tf)
      # print("Saving memory")
      # self.replay_buf.save(self.model_dir)
      # pickle_save(self.env_file, self.env)




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


  def _run_env():
    """Thread for running the environment. Must call `self._wait_train_done()`
    before selcting an action (by running the model). This ensures that the
    `self._train_model()` thread has finished the training step. After action
    is selected, it must call `self._signal_act_chosen()` to allow
    `self._train_model()` thread to start a new training step
    """
    raise NotImplementedError()


  def _train_model():
    """Thread for trianing the model. Must call `self._wait_act_chosen()`
    before trying to run a training step on the model. This ensures that the
    `self._run_env()` thread has finished selcting an action (by running the model).
    After training step is done, it must call `self._signal_train_done()` to allow
    `self._run_env()` thread to select a new action
    """
    raise NotImplementedError()


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
