import logging
import os
import tensorflow as tf

from rltf.envs.utils  import get_env_monitor
from rltf.utils       import seeding


logger = logging.getLogger(__name__)


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
               plots_layout=None,
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
      layout: dict or None. Used to configure the layout for video plots.
    """

    self.env            = env
    self.env_monitor    = get_env_monitor(env)

    self.batch_size     = batch_size
    self.model_dir      = os.path.join(model_dir, "tf/")
    self.save_freq      = save_freq
    self.log_freq       = log_freq
    self.restore_dir    = os.path.join(restore_dir, "tf/") if restore_dir is not None else None
    self.prng           = seeding.get_prng()

    self.start_step     = None          # Step from which the agent starts
    self.warm_up        = warm_up       # Step from which training starts
    self.stop_step      = stop_step     # Step at which training stops
    self.learn_started  = False         # Bool: Indicates if learning has started or not
    self.train_freq     = train_freq    # How often to run a training step

    self.eval_freq      = eval_freq     # How often to take an evaluation run
    self.eval_len       = eval_len      # How many steps to an evaluation run lasts

    self.layout         = plots_layout
    self.built          = False

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

    self.built = True
    if self.layout:
      self.plots_layout(self.layout)


  def plots_layout(self, layout):
    assert self.built
    self.env_monitor.conf_video_plots(layout=layout, train_tensors=self.model.plot_train,
      eval_tensors=self.model.plot_eval, plot_data=self.model.plot_data)


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


  def _get_feed_dict(self, batch, t):
    """Get the placeholder parameters to feed to the model while training
    Args:
      t: int. Current timestep
      batch: dict. Data to pack in feed_dict
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
