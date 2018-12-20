import json
import logging
import os
import re
import signal
import numpy as np
import tensorflow as tf

from rltf.utils import seeding


logger = logging.getLogger(__name__)


class Agent:
  """Base class for a Reinforcement Learning agent"""

  def __init__(self,
               eval_period,
               eval_len,
               model_dir,
               save_period=1000000,
               n_plays=0,
               load_model=None,
               load_regex=None,
               **model_kwargs
              ):
    """
    Args:
      eval_period: int. How many agent steps to take between every 2 evaluation runs
      eval_len: int. How many agent steps an evaluation run lasts. `<=0` means no evaluation
      model_dir: str. Directory path for the model, where logs and checkpoints will be saved. If the
        directory is not empty, the agent will restore the model continue training it
      n_plays: int. Number of separate play (or evaluation) runs to execute when `Agent.play()` is called.
        If `<=0`, `Agent.play()` will raise exception. If `>0`, `Agent.train()` will raise exception
      save_period: int. Save the model every `save_period` training steps. `<=0` means no saving
      load_model: str. Path to a directory which contains an existing model. The best_agent weights in
        this model will be loaded (no data in `load_model` will be overwritten)
      load_regex: str. Regular expression for matching variables whose values should be reused.
        If empty, all model variables are reused
      model_kwargs: dict. All uncaught arguments are automatically considered as arguments to be
        passed to the model
    """

    # Environment data
    self.env_train      = None
    self.env_eval       = None

    # Model data
    self.model          = None
    self.model_kwargs   = model_kwargs
    self.built          = False

    # Save and restore specs data
    self.model_dir      = model_dir
    self.reuse_model    = load_model
    self.reuse_regex    = None if load_regex is None else re.compile(load_regex)
    self.save_period    = save_period if save_period > 0 else np.inf
    self.train_saver    = None
    self.eval_saver     = None

    # Training data
    self.agent_step     = 0             # Current agent step
    self.prng           = seeding.get_prng()

    # Evaluation data
    self.eval_period    = eval_period   # How often to take an evaluation run
    self.eval_len       = eval_len      # How many steps to an evaluation run lasts
    self.eval_step      = 0             # Current agent eval step

    # Run data
    self.n_plays        = n_plays       # Number of evaluation runs in Agent.play()
    self.play_mode      = n_plays > 0   # True if Agent allowed to be run only in play mode
    self._terminate     = False         # Used to signal terminate. Must be monitored by the main loop

    # TensorFlow attributes
    self.sess           = None

    if not os.path.exists(self.tf_dir):
      os.makedirs(self.tf_dir)


  def build(self):
    """Build the graph. The graph is always built and initialized from scratch. After that, tf.Variables
     are restored (if needed). Meta graphs are not used. Calls `self._build()` and `self.model.build()`.
    - If `model_dir` is not empty, all TF variable values will be restored from the latest checkpoint
    - If `reuse_model is not None`, variables which match `reuse_regex` will be restored from the
      best agent checkpoint. The rest of the variables will retain their initialized random values
    """
    if self.built:
      return
    self.built = True

    if os.path.exists(self.state_file):
      restore = True
      assert not self.play_mode
      assert self.reuse_model is None
      assert self.reuse_regex is None
    else:
      restore = False

    # Set regex to match variables which should not be trained. Must be done before building the graph
    if self.reuse_model is not None and self.reuse_regex is not None:
      self.model.exclude_train_vars(self.reuse_regex)

    # Build the graph from scratch
    self._build_graph()

    # Restore all variables if model is being restored
    if restore:
      self._restore_vars()  # Restore tf variables
      self._restore()       # Execute agent-specific restore, e.g. restore buffer

    # Reuse some variables if model is being reused
    elif self.reuse_model is not None:
      self._reuse_vars()

    # NOTE: Create tf.train.Savers **after** building the whole graph
    # Create a saver for the training model
    self.train_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

    # Create a separate saver for the best agent
    self.eval_saver = tf.train.Saver(self.model.agent_vars, max_to_keep=1, save_relative_paths=True)


  def train(self):
    """Train the agent"""
    assert self.built, "You need to execute Agent.build() before calling Agent.train()"
    assert not self.play_mode      # Make sure in correct mode

    # Configure a safe exit
    self._configure_safe_exit()

    # If the agent was restored and it was previously terminated during an evaluation run,
    # complete this unfinished evaluation run before training starts
    if self.eval_period > 0 and self.agent_step % self.eval_period == 0:
      # Compute what eval step would be if eval run was able to complete
      eval_step = self.agent_step / self.eval_period * self.eval_len
      # Run evaluation if necessary
      if self.eval_step != eval_step:
        self._eval_agent()

    # Run the actual training process
    self._train()


  def play(self):
    """Evaluate the agent"""
    assert self.built, "You need to execute Agent.build() before calling Agent.play()"
    assert self.play_mode                       # Make sure in the correct mode and best agent restored
    assert self.eval_step == 0                  # Ensure a single call to eval
    assert self.reuse_regex is None             # Make sure all variables were restored

    # Configure a safe exit
    self._configure_safe_exit()

    # Run the actual evaluation process
    self._run_play()

    # Execute only agent-specific save, e.g. save statistics
    self._save()


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
    with open(self.state_file, 'r') as f:
      data = json.load(f)

    self.agent_step = data["train_step"]
    self.eval_step  = data["eval_step"]


  def _restore(self):
    """Execute agent-specific restore procedures"""
    pass


  def _reuse_vars(self):
    logger.info("Loading model '%s' and reusing variables", self.reuse_model)

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


  def _train(self):
    """Evaluate the agent. To be implemented by the inheriting class"""
    raise NotImplementedError()


  def _build(self):
    """Overload in subclasses to build additional TF objects. Do NOT call `self.model.build()`"""
    pass


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


  def _action_eval(self, state):
    """Return action selected by the agent for an evaluation step
    Args:
      state: np.array. Current state
    """
    raise NotImplementedError()


  def _eval_agent(self):
    """Subclass helper function.
    Execute a single evaluation run for the lenght of self.eval_len steps.
    """
    if self.eval_len <= 0 or self.eval_period <= 0:
      return

    logger.info("Starting evaluation run")

    # Compute the start and the end step
    start_step  = self.eval_step + 1
    stop_step   = start_step + self.eval_len

    # Reset the environment at the beginning
    obs = self.env_eval.reset()

    for t in range(start_step, stop_step):
      if self._terminate:
        break

      action = self._action_eval(obs)
      next_obs, rew, done, info = self.env_eval.step(action)

      # Reset the environment if end of episode
      if done:
        next_obs = self.env_eval.reset()
      obs = next_obs

    # Execute on successful loop completion
    else:
      best_agent = info["rltfmon.best_agent"]
      self._save_best_agent(best_agent)     # Save agent if the best so far
      self.eval_step = t                    # Update the eval step

      logger.info("Evaluation run finished")


  def _run_play(self):
    """Subclass helper function.
    Run the agent in evaluation mode. Should be called from `Agent._play()`
    """
    for _ in range(self.n_plays):
      self._eval_agent()


  def save(self):
    """Save the current agent status to disk. This function needs to be explicitly called.
    It is possible to implement automatic savers which save the data to disk at some period, but
    the agent state needs to be consistent, including train and eval steps, all model variables,
    possible monitors. Thus explicitly calling save is easier and more clear.

    Calls `self._save()` for any subclass specific save procedures.
    """
    if self.play_mode or not self._save_allowed():
      return

    logger.info("Saving the TF model and stats to %s", self.model_dir)

    # Save the model
    self.train_saver.save(self.sess, self.tf_dir, global_step=self.agent_step)

    # Execute additional agent-specific save proceudres
    self._save()

    # Save the agent state
    data = {
      "train_step": self.agent_step,
      "eval_step":  self.eval_step,
    }
    with open(self.state_file, 'w') as f:
      json.dump(data, f, indent=4, sort_keys=True)

    logger.info("Save finished successfully")


  def _save(self):
    """Override in subclasses in order to implement custom save procedures"""
    return


  def _save_allowed(self):
    """Return False if saving is not allowed at this point due to inconsistent state.
    To be overriden in subclasses."""
    return True


  def _save_best_agent(self, best_agent):
    """Save the best-performing agent.
    best_agent: bool. If True, the agent is the best so far. If False, do not save.
    """
    if not best_agent or self.play_mode:
      return

    save_dir = self.best_agent_dir
    logger.info("Saving best agent so far to %s", save_dir)
    # Save the model
    self.eval_saver.save(self.sess, save_dir, global_step=self.agent_step+1)
    logger.info("Save finished successfully")


  @property
  def state_file(self):
    return os.path.join(self.model_dir, "agent_state.json")


  @property
  def tf_dir(self):
    return os.path.join(self.model_dir, "tf/")


  @property
  def best_agent_dir(self):
    return os.path.join(self.model_dir, "tf/best_agent/")


  @property
  def reuse_ckpt(self):
    return self._ckpt_path(os.path.join(self.reuse_model, "tf/best_agent/"))


  @property
  def restore_ckpt(self):
    return self._ckpt_path(self.tf_dir)


  def _ckpt_path(self, ckpt_dir):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt is None:
      raise ValueError("No checkpoint found in {}".format(ckpt_dir))
    ckpt_path = ckpt.model_checkpoint_path #pylint: disable=no-member
    return ckpt_path


  @staticmethod
  def _get_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=no-member
    return tf.Session(config=config)


  def _configure_safe_exit(self):
    """Catch Ctrl+C in order to exit safely without interrupting the agent"""

    in_exit_call = False

    def safe_exit(*args, **kwargs): #pylint: disable=unused-argument
      nonlocal in_exit_call

      # If Ctrl+C pressed twice consecutively, then exit
      if in_exit_call or not self.built:
        import sys
        sys.exit(0)

      in_exit_call = True

      # Confirm that killing was intended
      y = ''
      while True:
        y = input("Do you really want to exit? [y/n]")
        if y not in ['y', 'n']:
          print("Response not recognized. Expected 'y' or 'n'.")
        else:
          break
      if y == 'n':
        logger.info("CONTINUING EXECUTION")
      else:
        logger.info("EXITING")
        self._terminate = True

      in_exit_call = False

    signal.signal(signal.SIGINT, safe_exit)
