import json
import logging
import os
import re
import tensorflow as tf

from rltf.envs.utils  import get_env_monitor
from rltf.utils       import seeding


logger = logging.getLogger(__name__)


class Agent:
  """Base class for a Reinforcement Learning agent"""

  def __init__(self,
               env_train,
               env_eval,
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
               confirm_kill=False,
               reuse_regex=None,
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
      plots_layout: dict or None. Used to configure the layout for video plots
      confirm_kill: bool. If True, you will be asked to confirm Ctrl+C
      reuse_regex: str or None. Regular expression for matching variables whose values should be reused.
        If None, all model variables are reused
    """

    self.env_train      = env_train
    self.env_train_mon  = get_env_monitor(env_train)
    self.env_eval       = env_eval
    self.env_eval_mon   = get_env_monitor(env_eval)

    self.batch_size     = batch_size
    self.model_dir      = model_dir
    self.save_freq      = save_freq
    self.log_freq       = log_freq
    self.restore_dir    = restore_dir
    self.prng           = seeding.get_prng()
    self.confirm_kill   = confirm_kill
    self.reuse_regex    = None if reuse_regex is None else re.compile(reuse_regex)

    self.warm_up        = warm_up       # Step from which training starts
    self.stop_step      = stop_step     # Step at which training stops
    self.learn_started  = False         # Bool: Indicates if learning has started or not
    self.train_freq     = train_freq    # How often to run a training step
    self.train_step     = 1             # Current agent train step

    self.eval_freq      = eval_freq     # How often to take an evaluation run
    self.eval_len       = eval_len      # How many steps to an evaluation run lasts
    self.eval_step      = 1             # Current agent eval step

    self.layout         = plots_layout
    self.built          = False

    # TensorFlow attributes
    self.model            = None
    self.summary          = None

    self.summary_op       = None

    self.sess             = None
    self.train_saver      = None
    self.eval_saver       = None
    self.tb_train_writer  = None
    self.tb_eval_writer   = None


  def build(self):
    """Build the graph. Automatically calls (in order) `self._build()` and `self.model.build()`
    or `self._restore()` and `self.model.restore()`.
    If `restore_dir is None`, the graph will be built and initialized from scratch.
    If `restore_dir == model_dir`, the graph will be restored and the build procedure will not be called.
    If `restore_dir != model_dir`, the graph will be built from scratch and initialized with the values.
    of the variables in `restore_dir` which match the provided pattern
    """

    restore = self.restore_dir is not None and self.restore_dir == self.model_dir
    reuse   = self.restore_dir is not None and self.restore_dir != self.model_dir

    if not restore:
      if reuse:
        # Set regex to match variables which should not be trained. Must be done before building the graph
        if self.reuse_regex is not None:
          self.model.exclude_train_vars(self.reuse_regex)
        # Build the model from scratch
        self._build_base()
        # Reuse variables
        self._reuse_base()
      else:
        # Build the model from scratch
        self._build_base()
    else:
      # Restore an existing model
      self._restore_base()

    # NOTE: Create tf.train.Saver **after** building the whole graph
    self.train_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

    # Create a separate saver for the best agent
    var_list = [v for v in self.model.variables if "agent_net" in v.name]
    self.eval_saver = tf.train.Saver(var_list, max_to_keep=1, save_relative_paths=True)

    # Create TensorBoard summary writers
    tb_dir = os.path.join(self.model_dir, "tf/tb/")
    self.tb_train_writer  = tf.summary.FileWriter(tb_dir, self.sess.graph)
    self.tb_eval_writer   = tf.summary.FileWriter(tb_dir)

    self.built = True
    if self.layout:
      self.plots_layout(self.layout)


  def plots_layout(self, layout):
    assert self.built
    self.env_train_mon.conf_video_plots(layout=layout, train_tensors=self.model.plot_train,
      eval_tensors=self.model.plot_eval, plot_data=self.model.plot_data)
    self.env_eval_mon.conf_video_plots(layout=layout, train_tensors=self.model.plot_train,
      eval_tensors=self.model.plot_eval, plot_data=self.model.plot_data)


  def train(self):
    raise NotImplementedError()


  def eval(self):
    raise NotImplementedError()


  def reset(self):
    """This method must be called at the end of every TRAINING episode. Allows for
    executing changes that stay the same for the duration of the whole episode.
    Returns:
      obs: np.array. The result of env_train.reset()
    """
    self._reset()
    self.model.reset(self.sess)
    return self.env_train.reset()


  def close(self):
    # Save before closing
    self.save()

    # Close the writers, the env and the session on exit
    self.tb_train_writer.close()
    self.tb_eval_writer.close()
    self.sess.close()
    self.env_train.close()
    self.env_eval.close()


  def _build_base(self):
    logger.info("Building model")

    # Call the subclass _build method
    self._build()

    # Build the model
    self.model.build()

    # Create an Op for all summaries
    self.summary_op = tf.summary.merge_all()

    # Create a session and initialize the model
    self.sess = self._get_sess()
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())

    # Initialize the model
    self.model.initialize(self.sess)


  def _restore_base(self):
    logger.info("Restoring model")

    # Get the checkpoint
    ckpt_path = self._ckpt_path()

    # Restore the graph structure
    saver = tf.train.import_meta_graph(ckpt_path + '.meta')
    graph = tf.get_default_graph()

    # Recover the model variables
    self.model.restore(graph)

    # Recover the agent subclass variables
    self._restore(graph)

    # Restore the values of all tf.Variables
    self.sess = self._get_sess()
    saver.restore(self.sess, ckpt_path)

    # Get the summary Op
    self.summary_op = graph.get_tensor_by_name("Merge/MergeSummary:0")

    # Recover the agent state
    state_file = os.path.join(self.model_dir, "agent_state.json")
    with open(state_file, 'r') as f:
      data = json.load(f)

    self.train_step = data["train_step"]
    self.eval_step  = data["eval_step"]
    self.learn_started = self.train_step >= self.warm_up

    if not self.learn_started:
      logger.warning("Training the restored model will not start immediately")
      logger.warning("Random policy will be run for %d steps", self.warm_up-self.train_step)


  def _reuse_base(self):
    logger.info("Reusing model variables:")

    # Get the list of variables to restore
    if self.reuse_regex is not None:
      var_list = [v for v in self.model.variables if self.reuse_regex.search(v.name)]
    else:
      var_list = self.model.variables

    # Log the variables being restored
    for v in var_list:
      logger.info(v.name)

    # Restore the variables
    saver = tf.train.Saver(var_list)
    saver.restore(self.sess, self._ckpt_path())


  def _ckpt_path(self):
    restore_dir = os.path.join(self.restore_dir, "tf/")
    ckpt = tf.train.get_checkpoint_state(restore_dir)
    if ckpt is None:
      raise ValueError("No checkpoint found in {}".format(restore_dir))
    ckpt_path = ckpt.model_checkpoint_path
    return ckpt_path


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


  def _run_train_step(self, t, run_summary):
    """Get the placeholder parameters to feed to the model while training
    Args:
      t: int. Current timestep
      run_summary: bool. Whether summary should be run and set during the train step
    """
    raise NotImplementedError()


  def _restore(self, graph):
    """Restore the Variables, placeholders and Ops needed by the class so that
    it can operate in exactly the same way as if `self._build()` was called
    Args:
      graph: tf.Graph. Graph, restored from a checkpoint
      resume: bool. True - training should continue as started. False - use only NN weights
    """
    raise NotImplementedError()


  def _define_log_info(self):
    custom_log_info = self._append_log_info()
    self.env_train_mon.define_log_info(custom_log_info)
    self.env_eval_mon.define_log_info(custom_log_info)


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


  def _log_stats(self, t, mode):
    """Log the training progress and append the TensorBoard summary.
    Note that the TensorBoard summary might be 1 step old.
    Args:
      t: int. Current timestep
      mode: str, either 't' (train) or 'e' (eval). The mode for which to log the statistics
    """
    if t % self.log_freq != 0:
      return
    assert mode in ['t', 'e']

    if mode == 't' and self.learn_started:
      # Log the statistics from the environment Monitor
      self.env_train_mon.log_stats(t)

      if self.summary:
        # Append the summary with custom data
        byte_summary = self.summary
        summary = tf.Summary()
        summary.ParseFromString(byte_summary)
        summary.value.add(tag="train/mean_ep_rew", simple_value=self.env_train_mon.mean_ep_rew)
        self._append_summary(summary, t)

        # Log with TensorBoard
        self.tb_train_writer.add_summary(summary, global_step=t)

    elif mode == 'e':
      # Log the statistics from the environment Monitor
      self.env_eval_mon.log_stats(t)

      # Add a TB summary
      summary = tf.Summary()
      summary.value.add(tag="eval/mean_ep_rew", simple_value=self.env_eval_mon.mean_ep_rew)
      summary.value.add(tag="eval/score",       simple_value=self.env_eval_mon.eval_score)
      self.tb_eval_writer.add_summary(summary, t)


  def save(self):
    if self.learn_started:
      logger.info("Saving the TF model and stats to %s", self.model_dir)

      # Save the monitor statistics
      self.env_train_mon.save()
      self.env_eval_mon.save()

      # Save the model
      model_dir = os.path.join(self.model_dir, "tf/")
      self.train_saver.save(self.sess, model_dir, global_step=self.train_step)

      self._save()

      # Save the agent state
      state_file = os.path.join(self.model_dir, "agent_state.json")
      data = {
        "train_step": self.train_step,
        "eval_step":  self.eval_step,
      }
      with open(state_file, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)

      # Flush the TB writers
      self.tb_train_writer.flush()
      self.tb_eval_writer.flush()

      logger.info("Save finished successfully")


  def _save(self):
    """Use by implementing class for custom save procedures"""
    return


  def _save_best(self, best_agent):
    """Save the best-performing agent.
    best_agent: bool. If True, the agent is the best so far. If False, do not save.
    """
    if not best_agent:
      return
    model_dir = os.path.join(self.model_dir, "tf/best_agent/")

    logger.info("Saving best agent so far to %s", model_dir)
    # Save the model
    self.eval_saver.save(self.sess, model_dir, global_step=self.train_step+1)
    logger.info("Save finished successfully")


  def _get_sess(self):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
