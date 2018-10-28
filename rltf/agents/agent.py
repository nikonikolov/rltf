import json
import logging
import os
import re
import tensorflow as tf

from rltf.utils import seeding


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
               save_freq=1000000,
               restore_dir=None,
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
      save_freq: int. Save the model every `save_freq` training steps. `<=0` means no saving
      restore_dir: str. Path to a directory which contains an existing model to restore. If
        `restore_dir==model_dir`, then training is continued from the saved model time step and
        the existing model is overwritten. Otherwise, the saved weights are used but training
        starts from step 0 and the model is saved in `model_dir`. If `None`, no restoring
      reuse_regex: str or None. Regular expression for matching variables whose values should be reused.
        If None, all model variables are reused
    """

    # Environment data
    self.env_train      = env_train
    self.env_eval       = env_eval

    # Model data
    self.model          = None
    self.built          = False

    # Save and restore specs data
    self.model_dir      = model_dir
    self.restore_dir    = restore_dir
    self.reuse_regex    = None if reuse_regex is None else re.compile(reuse_regex)
    self.save_freq      = save_freq
    self.train_saver    = None
    self.eval_saver     = None

    # Training data
    self.warm_up        = warm_up       # Step from which training starts
    self.stop_step      = stop_step     # Step at which training stops
    self.learn_started  = False         # Bool: Indicates if learning has started or not
    self.train_freq     = train_freq    # How often to run a training step
    self.train_step     = 0             # Current agent train step
    self.batch_size     = batch_size
    self.prng           = seeding.get_prng()

    # Evaluation data
    self.eval_freq      = eval_freq     # How often to take an evaluation run
    self.eval_len       = eval_len      # How many steps to an evaluation run lasts
    self.eval_step      = 0             # Current agent eval step

    # TensorFlow attributes
    self.sess           = None

    if not os.path.exists(self.tf_dir):
      os.makedirs(self.tf_dir)


  def build(self):
    """Build the graph. The graph is always built and initialized from scratch. After that, tf.Variables
     are restored (if needed). Meta graphs are not used. Calls `self._build()` and `self.model.build()`.
    If `restore_dir is None`, the graph will be initialized from scratch.
    If `restore_dir == model_dir`, all graph variable values will be restored from checkpoint
    If `restore_dir != model_dir`, variables which match the provided pattern will be restored from
    checkpoint. The rest of the variables will retain their original random values
    """
    if self.built:
      return
    self.built = True

    restore = self.restore_dir is not None and self.restore_dir == self.model_dir
    reuse   = self.restore_dir is not None and self.restore_dir != self.model_dir

    # Set regex to match variables which should not be trained. Must be done before building the graph
    if reuse and self.reuse_regex is not None:
      self.model.exclude_train_vars(self.reuse_regex)

    # Build the graph from scratch
    self._build_graph()

    # Restore all variables if model is being restored
    if restore:
      self._restore_vars()  # Restore tf variables
      self._restore()       # Execute agent-specific restore, e.g. restore buffer

    # Reuse some variables if model is being reused
    elif reuse:
      self._reuse_vars()

    # NOTE: Create tf.train.Savers **after** building the whole graph
    # Create a saver for the training model
    self.train_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

    # Create a separate saver for the best agent
    var_list = [v for v in self.model.variables if "agent_net" in v.name]
    self.eval_saver = tf.train.Saver(var_list, max_to_keep=1, save_relative_paths=True)


  def train(self):
    """Train the agent"""
    raise NotImplementedError()


  def eval(self):
    """Evaluate the agent"""
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
    self.sess.close()
    self.env_train.close()
    self.env_eval.close()


  def _build_graph(self):
    logger.info("Building model")

    # Call the subclass _build method
    self._build()

    # Build the model
    self.model.build()

    # Create a session and initialize the model
    self.sess = self._get_sess()
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())

    # Initialize the model
    self.model.initialize(self.sess)


  def _restore_vars(self):
    logger.info("Restoring model")

    # Restore all variables
    saver = tf.train.Saver()
    saver.restore(self.sess, self.restore_ckpt)

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


  def _restore(self):
    """Execute agent-specific restore procedures"""
    pass


  def _reuse_vars(self):
    logger.info("Reusing model variables:")

    # Get the list of variables to restore
    if self.reuse_regex is not None:
      var_list = [v for v in self.model.variables if self.reuse_regex.search(v.name)]
    else:
      var_list = self.model.variables

    # Log the variables being restored
    for v in var_list:
      logger.info(v.name)

    # Restore the best agent variables
    saver = tf.train.Saver(var_list)
    saver.restore(self.sess, self.reuse_ckpt)


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


  def _run_train_step(self, t):
    """Get the placeholder parameters to feed to the model while training
    Args:
      t: int. Current timestep
    """
    raise NotImplementedError()


  def save(self):
    """Save the current agent status to disk. This function needs to be explicitly called.
    It is possible to implement automatic savers which save the data to disk at some period, but
    the agent state needs to be consistent, including train and eval steps, all model variables,
    possible monitors. Thus explicitly calling save is easier and more clear.

    Calls `self._save()` for any subclass specific save procedures.
    """
    if not self.learn_started:
      return

    logger.info("Saving the TF model and stats to %s", self.model_dir)

    # Save the model
    self.train_saver.save(self.sess, self.tf_dir, global_step=self.train_step)

    # Execute additional agent-specific save proceudres
    self._save()

    # Save the agent state
    state_file = os.path.join(self.model_dir, "agent_state.json")
    data = {
      "train_step": self.train_step,
      "eval_step":  self.eval_step,
    }
    with open(state_file, 'w') as f:
      json.dump(data, f, indent=4, sort_keys=True)

    logger.info("Save finished successfully")


  def _save(self):
    """Overload in subclasses in order to implement custom save procedures"""
    return


  def _save_best_agent(self, best_agent):
    """Save the best-performing agent.
    best_agent: bool. If True, the agent is the best so far. If False, do not save.
    """
    if not best_agent:
      return

    save_dir = self.best_agent_dir
    logger.info("Saving best agent so far to %s", save_dir)
    # Save the model
    self.eval_saver.save(self.sess, save_dir, global_step=self.train_step+1)
    logger.info("Save finished successfully")


  @property
  def tf_dir(self):
    return os.path.join(self.model_dir, "tf/")


  @property
  def best_agent_dir(self):
    return os.path.join(self.model_dir, "tf/best_agent/")


  @property
  def reuse_ckpt(self):
    return self._ckpt_path(self.best_agent_dir)


  @property
  def restore_ckpt(self):
    return self._ckpt_path(self.tf_dir)


  def _ckpt_path(self, ckpt_dir):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt is None:
      raise ValueError("No checkpoint found in {}".format(ckpt_dir))
    ckpt_path = ckpt.model_checkpoint_path
    return ckpt_path


  @staticmethod
  def _get_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
