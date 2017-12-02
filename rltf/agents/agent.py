import numpy as np
import os
import tensorflow as tf
import threading
import sys

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
               save_freq=1e5,
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

    self.mean_ep_rew       = -float('nan')
    self.best_mean_ep_rew  = -float('inf')


  def build(self):
    raise NotImplementedError()


  def train(self):
    raise NotImplementedError()


  def reset(self):
    raise NotImplementedError()

  def close(self):
    # Close session on exit
    self.tb_writer.close()
    self.sess.close()


  def _build(self):
    """Build general logging placeholders and timestep variable (needed by any 
    agent). Before calling this, self.model.build() must have been called and
    all summary operators must have been added.
    """

    # Build the model network
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

    # Create the Saver object: NOTE that you must do it after building the whole graph
    self.saver     = tf.train.Saver()
    self.tb_writer = tf.summary.FileWriter(self.model_dir + "tb/", self.sess.graph)

    # Initialize the model
    self.model.initialize(self.sess)


  def _restore(self, ckpt):
    raise NotImplementedError()

    # print("Restoring model")
    # # Restore the graph
    # self._restore(ckpt)
    
    # ckpt_path = ckpt.model_checkpoint_path
    # g_saver   = self.model.restore_graph(ckpt_path)

    # # Restore the session
    # self.sess = tf.Session()
    # g_saver.restore(self.sess, ckpt.model_checkpoint_path)
    # # Make sure no variables are not initialized
    # # assert not tf.report_uninitialized_variables()

    # # Get variables and placeholders
    # self.t_tf                 = tf.get_default_graph().get_tensor_by_name("t_tf:0")
    # self.t_tf_inc             = tf.get_default_graph().get_operation_by_name("t_tf_inc")
    # self.mean_ep_rew_ph       = tf.get_default_graph().get_tensor_by_name("mean_ep_rew_ph:0")
    # self.best_mean_ep_rew_ph  = tf.get_default_graph().get_tensor_by_name("best_mean_ep_rew_ph:0")

    # # Get the summary op
    # self.summary_op           = tf.get_default_graph().get_tensor_by_name("Merge/MergeSummary:0")

    # # Set control variables
    # self.start_step     = self.sess.run(self.t_var)
    # self.learn_started  = True



    # # Restore the buffer
    # self.replay_buf        = self.replay_buf.restore(self.model_dir)


  def _build_log_list(self, log_info):
    default_info = []
    default_info.append(("timestep",              "%d", lambda t: t))
    default_info.append(("episodes",              "%d", lambda t: self.episodes))
    default_info.append(("mean reward (100 eps)", "%f", lambda t: self.mean_ep_rew))
    default_info.append(("best mean reward",      "%f", lambda t: self.best_mean_ep_rew))

    log_info = default_info + log_info

    str_sizes = [len(s) for s, _, _ in log_info]
    pad = max(str_sizes) + 2

    self.log_list  = [(s.ljust(pad) + ptype, v) for s, ptype, v in log_info]


  def _log_progress(self, t):
    """Log the training progress and append the TensorBoard summary.
    Note that the TensorBoard summary might be 1 step older.
    
    Args:
      t: int. Current timestep
    """

    episode_rewards = self.env_monitor.get_episode_rewards()
    if len(episode_rewards) > 0:
      self.mean_ep_rew      = np.mean(episode_rewards[-100:])
      self.best_mean_ep_rew = max(self.best_mean_ep_rew, self.mean_ep_rew)
    self.episodes = len(episode_rewards)

    if t % self.log_freq == 0 and self.learn_started:
      for s, lambda_v in self.log_list:
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
