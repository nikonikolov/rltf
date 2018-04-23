import logging
import os
import threading

import tensorflow as tf

from rltf.conf        import STATS_LOGGER_NAME
from rltf.envs.utils  import get_env_monitor

stats_logger  = logging.getLogger(STATS_LOGGER_NAME)
logger        = logging.getLogger(__name__)


class Agent:
  """Base class for a Reinforcement Learning agent"""

  def __init__(self,
               env,
               train_freq,
               warm_up,
               stop_step,
               eval_freq,
               eval_len,
               batch_size,
               model_dir,
               log_freq=10000,
               save_freq=100000,
               restore_dir=None,
              ):
    """
    Args:
      env: gym.Env. Environment in which the model will be trained.
      train_freq: int. How many environment actions to take between every 2 learning steps
      warm_up: int. Number of random steps before training starts
      stop_step: int. Training step at which learning stops
      eval_freq: int. How many agent steps to take between every 2 evaluation runs
      eval_len: int. How many agent steps an evaluation run lasts. `<=0` means no evaluation
      batch_size: int. Batch size for training the model
      model_dir: string. Directory path for the model logs and checkpoints
      log_freq: int. Add TensorBoard summary and print progress every log_freq agent steps
      save_freq: int. Save the model every `save_freq` training steps. `<=0` means no saving
      restore_dir: str. Path to a directory which contains an existing model to restore. If
        `restore_dir==model_dir`, then training is continued from the saved model time step and
        the existing model is overwritten. Otherwise, the saved weights are used but training
        starts from step 0 and the model is saved in `model_dir`. If `None`, no restoring
    """

    self.env            = env
    self.env_monitor    = get_env_monitor(env)

    self.batch_size     = batch_size
    self.model_dir      = os.path.join(model_dir, "tf/")
    self.save_freq      = save_freq
    self.log_freq       = log_freq
    self.restore_dir    = os.path.join(restore_dir, "tf/") if restore_dir is not None else None

    self.start_step     = None          # Step from which the agent starts
    self.warm_up        = warm_up       # Step from which training starts
    self.stop_step      = stop_step     # Step at which training stops
    self.learn_started  = False         # Bool: Indicates if learning has started or not
    self.train_freq     = train_freq    # How often to run a training step

    self.eval_freq      = eval_freq     # How often to take an evaluation run
    self.eval_len       = eval_len      # How many steps to an evaluation run lasts

    # TensorFlow attributes
    self.model            = None
    self.summary          = None

    self.t_train          = None
    self.t_train_inc      = None
    self.t_eval           = None
    self.t_eval_inc       = None
    self.summary_op       = None

    self.sess             = None
    self.saver            = None
    self.tb_train_writer  = None
    self.tb_eval_writer   = None


  def build(self):
    """Build the graph. If `restore_dir is not None`, the graph will be restored from
    `restore_dir`. Automatically calls (in order) `self._build()` and `self.model.build()`
    or `self._restore()` and `self.model.restore()`.
    """

    # Build the model from scratch
    if self.restore_dir is None:
      self._build_base()
    # Restore an existing model
    else:
      self._restore_base()

    # NOTE: Create the tf.train.Saver  **after** building the whole graph
    self.saver     = tf.train.Saver(max_to_keep=2, save_relative_paths=True)
    # Create TensorBoard summary writers
    self.tb_train_writer  = tf.summary.FileWriter(self.model_dir + "tb/", self.sess.graph)
    self.tb_eval_writer   = tf.summary.FileWriter(self.model_dir + "tb/")


  def train(self):
    raise NotImplementedError()


  def eval(self):
    raise NotImplementedError()


  def reset(self):
    """This method must be called at the end of every episode. Allows for
    executing changes that stay the same for the duration of the whole episode.
    Note that it gets called both in train and eval mode
    Returns:
      obs: np.array. The result of self.env.reset()
    """
    if self.learn_started:
      self.model.reset(self.sess)
      self._reset()
    return self.env.reset()


  def close(self):
    # Save before closing
    self.save()

    # Close the writers, the env and the session on exit
    self.tb_train_writer.close()
    self.tb_eval_writer.close()
    self.sess.close()
    self.env.close()


  def _build_base(self):
    logger.info("Building model")

    # Call the subclass _build method
    self._build()

    # Build the model
    self.model.build()

    # Create timestep variable
    with tf.device('/cpu:0'):
      self.t_train     = tf.Variable(0, dtype=tf.int32, trainable=False, name="t_train")
      self.t_eval      = tf.Variable(0, dtype=tf.int32, trainable=False, name="t_eval")
      self.t_train_inc = tf.assign_add(self.t_train, 1, name="t_train_inc")
      self.t_eval_inc  = tf.assign_add(self.t_eval,  1, name="t_eval_inc")

      # Create an Op for all summaries
      self.summary_op = tf.summary.merge_all()

    # Set control variables
    self.start_step = 1

    # Create a session and initialize the model
    self.sess = self._get_sess()
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())

    # Initialize the model
    self.model.initialize(self.sess)


  def _restore_base(self):

    resume = self.restore_dir == self.model_dir

    logger.info("%s model", "Resuming" if resume else "Reusing")

    # Get the checkpoint
    ckpt = tf.train.get_checkpoint_state(self.restore_dir)
    if ckpt is None:
      raise ValueError("No checkpoint found in {}".format(self.restore_dir))
    ckpt_path = ckpt.model_checkpoint_path

    # Restore the graph structure
    saver = tf.train.import_meta_graph(ckpt_path + '.meta')
    graph = tf.get_default_graph()

    # Recover the train and eval step variables
    global_vars       = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.t_train      = [v for v in global_vars if v.name == "t_train:0"][0]
    self.t_eval       = [v for v in global_vars if v.name == "t_eval:0"][0]
    self.t_train_inc  = graph.get_operation_by_name("t_train_inc")
    self.t_eval_inc   = graph.get_operation_by_name("t_eval_inc")

    # Recover the model variables
    self.model.restore(graph)

    # Recover the agent subclass variables
    self._restore(graph)

    # Restore the values of all tf.Variables
    self.sess = self._get_sess()
    saver.restore(self.sess, ckpt_path)

    # Get the summary Op
    self.summary_op = graph.get_tensor_by_name("Merge/MergeSummary:0")

    # Set control variables
    if resume:
      self.start_step = self.sess.run(self.t_train)
      # Ensure that you have enough random experience before training starts
      self.warm_up    = self.start_step + self.warm_up
    else:
      # Reset all step variables
      self.start_step = 1
      self.sess.run(tf.assign(self.t_train, 0))
      self.sess.run(tf.assign(self.t_eval,  0))


  def _build(self):
    """Used by the subclass to build class specific TF objects. Must not call
    `self.model.build()`
    """
    raise NotImplementedError()


  def _reset(self):
    """Reset method to be implemented by the inheriting class"""
    raise NotImplementedError()


  def _get_feed_dict(self, t):
    """Get the placeholder parameters to feed to the model while training
    Args:
      t: int. Current timestep
    """

    raise NotImplementedError()


  def _restore(self, graph):
    """Restore the Variables, placeholders and Ops needed by the class so that
    it can operate in exactly the same way as if `self._build()` was called

    Args:
      graph: tf.Graph. Graph, restored from a checkpoint
    """
    raise NotImplementedError()


  def _define_log_info(self):
    custom_log_info = self._append_log_info()
    self.env_monitor.define_log_info(custom_log_info)


  def _append_log_info(self):
    """
    Returns:
      List of tuples `(name, format, lambda)` with information of custom subclass
      parameters to log during training. `name`: `str`, the name of the reported
      value. `modifier`: `str`, the type modifier for printing the value.
      `lambda`: A function that takes the current timestep as argument and
      returns the value to be printed.
    """
    raise NotImplementedError()


  def _append_summary(self, summary, t):
    """Append the tf.Summary that will be written to disk at timestep t with custom data.
    Used only in train mode
    Args:
      summary: tf.Summary. The summary to append
      t: int. Current time step
    """
    raise NotImplementedError()


  def _log_stats(self, t):
    """Log the training progress and append the TensorBoard summary.
    Note that the TensorBoard summary might be 1 step old.
    Args:
      t: int. Current timestep
    """

    # Log the statistics from the environment Monitor
    if t % self.log_freq == 0 and self.learn_started:
      self.env_monitor.log_stats(t)

      if self.summary:
        # Append the summary with custom data
        byte_summary = self.summary
        summary = tf.Summary()
        summary.ParseFromString(byte_summary)
        summary.value.add(tag="train/mean_ep_rew", simple_value=self.env_monitor.get_mean_ep_rew())
        self._append_summary(summary, t)

        # Log with TensorBoard
        self.tb_train_writer.add_summary(summary, global_step=t)


  def save(self):
    if self.learn_started:
      dirname = os.path.dirname(os.path.dirname(self.model_dir))
      logger.info("Saving the TF model and stats to %s", dirname)

      # Save the monitor statistics
      self.env_monitor.save()

      # Save the model
      self.saver.save(self.sess, self.model_dir, global_step=self.t_train)

      # Flush the TB writers
      self.tb_train_writer.flush()
      self.tb_eval_writer.flush()

      logger.info("Save finished successfully")


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
    self._act_chosen = threading.Event()
    self._train_done = threading.Event()

    self._terminate  = False


  def train(self):

    self._act_chosen.clear()
    self._train_done.set()

    env_thread  = threading.Thread(name='env_thread', target=self._run_env)
    nn_thread   = threading.Thread(name='net_thread', target=self._train_model)

    nn_thread.start()
    env_thread.start()

    # Wait for threads
    try:
      env_thread.join()
      nn_thread.join()
    except KeyboardInterrupt:
      logger.info("EXITING")
      self._terminate = True
      env_thread.join()
      nn_thread.join()


  def eval(self):

    logger.info("Starting evaluation")

    # Set the monitor in evaluation mode
    self.env_monitor.mode = 'e'

    start_step  = self.sess.run(self.t_eval) + 1
    stop_step   = start_step + self.eval_len
    stop_step   = stop_step - stop_step % self.eval_len + 1   # Restore point might be the middle of eval

    obs = self.reset()

    for t in range (start_step, stop_step):
      if self._terminate:
        break

      # Increment the current eval step
      self.sess.run(self.t_eval_inc)

      action = self._action_eval(obs, t)
      next_obs, _, done, _ = self.env.step(action)

      # Reset the environment if end of episode
      if done:
        next_obs = self.reset()
      obs = next_obs

      if t % self.log_freq == 0:
        # Log the statistics
        self.env_monitor.log_stats(t)

        # Add a TB summary
        summary = tf.Summary()
        summary.value.add(tag="eval/mean_ep_rew", simple_value=self.env_monitor.get_mean_ep_rew())
        self.tb_eval_writer.add_summary(summary, t)

    # Set the monitor back to train mode
    self.env_monitor.mode = 't'

    logger.info("Evaluation finished")


  def _action_train(self, state, t):
    """Return action selected by the agent for a training step
    Args:
      state: np.array. Current state
      t: int. Current timestep
    """
    raise NotImplementedError()


  def _action_eval(self, state, t):
    """Return action selected by the agent for an evaluation step
    Args:
      state: np.array. Current state
      t: int. Current timestep
    """
    raise NotImplementedError()


  def _run_env(self):
    """Thread for running the environment. Must call `self._wait_train_done()`
    before selcting an action (by running the model). This ensures that the
    `self._train_model()` thread has finished the training step. After action
    is selected, it must call `self._signal_act_chosen()` to allow
    `self._train_model()` thread to start a new training step
    """

    obs = self.reset()

    for t in range (self.start_step, self.stop_step+1):
      if self._terminate:
        self._signal_act_chosen()
        break

      # Stop and run evaluation procedure
      if self.eval_len > 0 and t % self.eval_freq == 0:
        self.eval()
        # Reset the environment on return
        obs = self.reset()

      # Get an action to run
      if self.learn_started:
        try:
          action = self._action_train(obs, t)
        except tf.errors.InvalidArgumentError as e:
          logger.exception("Numerical Exception: %s", e)
          action = self.env.action_space.sample()
          # Terminate program
          self._terminate = True

      # Choose random action if learning has not started
      else:
        action = self.env.action_space.sample()

      # Signal to net_thread that action is chosen
      self._signal_act_chosen()

      # Increment the TF timestep variable
      self.sess.run(self.t_train_inc)

      # Run action
      next_obs, reward, done, _ = self.env.step(action)

      # Store the effect of the action taken upon obs
      self.replay_buf.store(obs, action, reward, done)

      self._log_stats(t)

      # Wait until net_thread is done
      self._wait_train_done()

      # Reset the environment if end of episode
      if done:
        next_obs = self.reset()
      obs = next_obs


  def _train_model(self):
    """Thread for trianing the model. Must call `self._wait_act_chosen()`
    before trying to run a training step on the model. This ensures that the
    `self._run_env()` thread has finished selcting an action (by running the model).
    After training step is done, it must call `self._signal_train_done()` to allow
    `self._run_env()` thread to select a new action
    """

    for t in range (self.start_step, self.stop_step+1):
      if self._terminate:
        self._signal_train_done()
        break

      if (t >= self.warm_up and t % self.train_freq == 0):

        self.learn_started = True

        # Compose feed_dict
        feed_dict = self._get_feed_dict(t)

        self._wait_act_chosen()

        # Run a training step
        if t % self.log_freq + self.train_freq >= self.log_freq:
          self.summary, _ = self.sess.run([self.summary_op, self.model.train_op], feed_dict=feed_dict)
        else:
          self.sess.run(self.model.train_op, feed_dict=feed_dict)

        # Update target network
        if t % self.update_target_freq == 0:
          self.sess.run(self.model.update_target)

      else:
        self._wait_act_chosen()

      if self.save_freq > 0 and t % self.save_freq == 0:
        self.save()

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
    # Signal to env thread that the training step is done running
    self._train_done.set()
