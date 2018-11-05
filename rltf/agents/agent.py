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
               train_period,
               warm_up,
               stop_step,
               eval_period,
               eval_len,
               batch_size,
               model_dir,
               save_period=1000000,
               n_evals=0,
               load_model=None,
               load_regex=None,
              ):
    """
    Args:
      env: gym.Env. Environment in which the model will be trained.
      train_period: int. How many environment actions to take between every 2 learning steps
      warm_up: int. Number of random steps before training starts
      stop_step: int. Training step at which learning stops
      eval_period: int. How many agent steps to take between every 2 evaluation runs
      eval_len: int. How many agent steps an evaluation run lasts. `<=0` means no evaluation
      batch_size: int. Batch size for training the model
      model_dir: str. Directory path for the model, where logs and checkpoints will be saved. If the
        directory is not empty, the agent will restore the model continue training it
      n_evals: int. Number of separate evaluation runs to execute when `Agent.eval()` is called. If `<=0`,
        `Agent.eval()` will raise exception. If `>0`, `Agent.train()` will raise exception
      save_period: int. Save the model every `save_period` training steps. `<=0` means no saving
      load_model: str. Path to a directory which contains an existing model. The best_agent weights in
        this model will be loaded (no data in `load_model` will be overwritten)
      load_regex: str. Regular expression for matching variables whose values should be reused.
        If empty, all model variables are reused
    """

    # Environment data
    self.env_train      = env_train
    self.env_eval       = env_eval

    # Model data
    self.model          = None
    self.built          = False

    # Save and restore specs data
    self.model_dir      = model_dir
    self.reuse_model    = load_model
    self.reuse_regex    = None if load_regex is None else re.compile(load_regex)
    self.save_period    = save_period
    self.train_saver    = None
    self.eval_saver     = None

    # Training data
    self.warm_up        = warm_up       # Step from which training starts
    self.stop_step      = stop_step     # Step at which training stops
    self.learn_started  = False         # Bool: Indicates if learning has started or not
    self.train_period   = train_period  # How often to run a training step
    self.train_step     = 0             # Current agent train step
    self.batch_size     = batch_size
    self.prng           = seeding.get_prng()

    # Evaluation data
    self.eval_period    = eval_period   # How often to take an evaluation run
    self.eval_len       = eval_len      # How many steps to an evaluation run lasts
    self.eval_step      = 0             # Current agent eval step
    self.n_evals        = n_evals       # Number of evaluation runs in Agent.eval()

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
      assert self.n_evals <= 0
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
    var_list = [v for v in self.model.variables if "agent_net" in v.name]
    self.eval_saver = tf.train.Saver(var_list, max_to_keep=1, save_relative_paths=True)


  def train(self):
    """Train the agent"""
    assert self.built, "You need to execute Agent.build() before calling Agent.train()"
    assert self.n_evals <= 0      # Make sure in correct mode

    # If the agent was restored and it was previously terminated during an evaluation run,
    # complete this unfinished evaluation run before training starts
    if self.eval_period > 0 and self.train_step % self.eval_period == 0:
      # Compute what eval step would be if eval run was able to complete
      eval_step = self.train_step / self.eval_period * self.eval_len
      # Run evaluation if necessary
      if self.eval_step != eval_step:
        self._eval_agent()

    # Run the actual training process
    self._train()


  def eval(self):
    """Evaluate the agent"""
    assert self.built, "You need to execute Agent.build() before calling Agent.eval()"
    assert self.n_evals > 0                     # Make sure in the correct mode and best agent restored
    assert self.eval_step == 0                  # Ensure a single call to eval
    assert self.reuse_regex is None             # Make sure all variables were restored

    # Run the actual evaluation process
    self._eval()

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


  def _eval(self):
    """Evaluate the agent. To be implemented by the inheriting class. Can call Agent._run_eval()"""
    raise NotImplementedError()


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


  def _action_eval(self, state, t):
    """Return action selected by the agent for an evaluation step
    Args:
      state: np.array. Current state
      t: int. Current timestep
    """
    raise NotImplementedError()


  def _check_terminate(self):
    """Check if train or eval loop must terminate because of Ctrl+C"""
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
      if self._check_terminate():
        break

      action = self._action_eval(obs, t)
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


  def _run_eval(self):
    """Subclass helper function.
    Run the agent in evaluation mode. Should be called from `Agent._eval()`
    """
    for _ in range(self.n_evals):
      self._eval_agent()


  def save(self):
    """Save the current agent status to disk. This function needs to be explicitly called.
    It is possible to implement automatic savers which save the data to disk at some period, but
    the agent state needs to be consistent, including train and eval steps, all model variables,
    possible monitors. Thus explicitly calling save is easier and more clear.

    Calls `self._save()` for any subclass specific save procedures.
    """
    if not self.learn_started or self.n_evals > 0:
      return

    logger.info("Saving the TF model and stats to %s", self.model_dir)

    # Save the model
    self.train_saver.save(self.sess, self.tf_dir, global_step=self.train_step)

    # Execute additional agent-specific save proceudres
    self._save()

    # Save the agent state
    data = {
      "train_step": self.train_step,
      "eval_step":  self.eval_step,
    }
    with open(self.state_file, 'w') as f:
      json.dump(data, f, indent=4, sort_keys=True)

    logger.info("Save finished successfully")


  def _save(self):
    """Overload in subclasses in order to implement custom save procedures"""
    return


  def _save_best_agent(self, best_agent):
    """Save the best-performing agent.
    best_agent: bool. If True, the agent is the best so far. If False, do not save.
    """
    if not best_agent or self.n_evals > 0:
      return

    save_dir = self.best_agent_dir
    logger.info("Saving best agent so far to %s", save_dir)
    # Save the model
    self.eval_saver.save(self.sess, save_dir, global_step=self.train_step+1)
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
    ckpt_path = ckpt.model_checkpoint_path
    return ckpt_path


  @staticmethod
  def _get_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
