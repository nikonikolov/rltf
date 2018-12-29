import json
import logging
import os
import time
import numpy as np
import tensorflow as tf

from gym.utils  import atomic_write

from rltf.utils import rltf_conf
from rltf.utils import rltf_log


logger        = logging.getLogger(__name__)
stats_logger  = logging.getLogger(rltf_conf.STATS_LOGGER_NAME)

# Constant for number of most recent episodes over which to report some of the runtime statistitcs
N_EPS_STATS   = 100


class StatsRecorder:

  def __init__(self, log_dir, mode, log_period=None, eval_period=None):
    """
    Args:
      log_dir: str. The path for the directory where the videos are saved
      mode: str. Either 't' (for train) or `e` (for eval)
      log_period: int. The period for logging statistics. If provided, statistics are logged automatically.
        Otherwise, log_stats() has to be called from outside. If mode == 'e', then it must be provided and
        must equal the evaluation length in order to keep correct evaluation statistics
      eval_period: int. Required only in evaluation mode. Needed to compute the correct logging step.
    """

    if log_period is not None:
      assert log_period > 0
      autolog = True
    else:
      assert mode != 'e', "'log_period' must be provided in evaluation mode"
      log_period = 1
      autolog = False

    # Member data
    self.log_dir      = os.path.join(log_dir, "data")
    self.tb_dir       = os.path.join(log_dir, "tb/")
    self.autolog      = autolog
    self.log_period   = log_period
    self._mode        = mode      # Running mode: either 't' (train) or 'e' (eval)
    self.n_episodes   = N_EPS_STATS if log_period is not None else None
    self.eval_period  = eval_period

    # Stdout data
    self.log_spec     = None    # Tuples containing the specification for fetching stdout data
    self.stdout_set   = False   # Track whether log_spec has been compiled

    # TensorBoard data
    suffix            = ".train" if self.mode == 't' else ".eval"
    self.tb_writer    = tf.summary.FileWriter(self.tb_dir, filename_suffix=suffix)
    self.summary_dict = None
    self.summary      = None
    self._get_summary = self._default_summary_getter  # Function that fetches the summary


    # Status
    # NOTE: self._env_eps is the episode ID, counted based on calls to env.unwrapped.reset().
    # This value might be different from `len(self.eps_rews)`, e.g. if the environment is
    # reset in the middle of an episode, `len(self.eps_rews)` will not change
    self._agent_steps  = 0    # Total number of agent steps so far
    self._env_steps    = 0    # Total number of env steps so far
    self._agent_eps    = 0    # Number of agent calls to env.reset()
    self._env_eps      = 0    # Number of calls to env.unwrapped.reset()

    # Statistics
    # NOTE: stats_steps and stats_inds are needed to be able to isolate and plot performance afterwards
    # In eval mode:
    #   - stats_steps corresponds to the agent step after the end of each evaluation run
    #     (counted in evaluation steps, NOT agent train steps)
    #   - stats_inds corresponds to the total number of complete evaluation episodes
    #     after the end of each evaluation run
    # In train mode:
    #   - stats_steps corresponds to the agent train step at each logging report
    #   - stats_inds corresponds to the total number of complete train episodes at each logging report
    self.ep_rews      = []    # List of the cumulative returns of all environment episodes
    self.ep_lens      = []    # List of the lengths of all environment episodes
    self.stats_steps  = []    # The agent step at each logging event
    self.stats_inds   = []    # The number of env episodes at each logging event
    self.stats        = None  # A dictionary with runtime statistics
    self.active       = False # Track whether env.step and env.reset() were executed via this monitor

    # Episode tracking
    self.ep_reward  = None      # Track the episode reward so far
    self.step_rew   = None      # Track the total reward from the current agent step
    self.ep_steps   = None      # Track the episode environment steps so far
    self.env_done   = None      # Track whether the environment is done

    # Initialize self.stats and the default self.log_spec
    self._init_stats()
    self._init_stdout()

    # Create log_dir or resume with saved data if log_dir exists
    if not os.path.exists(self.log_dir):
      logger.info('Creating stats directory %s', self.log_dir)
      os.makedirs(self.log_dir)
    elif len(os.listdir(self.log_dir)) > 0:
      logger.info('Resuming with stats data from %s', self.log_dir)
      self._resume()


  def before_agent_step(self, action):
    self.active = True
    self._agent_steps += 1
    # Keep this here. Sometimes, env.reset() might induce env.step(), but not
    # agent.env.step() due to Wrapper behavior
    self.step_rew = 0


  #pylint: disable=unused-argument
  def after_agent_step(self, obs, reward, done, info):
    # Append stats data for the agent
    info["rltfmon.ep_rew"]   = self.ep_reward
    info["rltfmon.step_rew"] = self.step_rew

    # NOTE:
    # - Logging needs to happen here, NOT in after_env_step. The latter might see the same value of
    #   self._agent_steps in more than one call and output several logging messages with the same data
    # - Logging TensorBoard summary data here does not interfere with any agent training step, even
    #   if it is concurrently executed, as long as summary has been updated sufficiently recently
    if self.autolog:
      self.log_stats(info)
    self.active = False


  def before_agent_reset(self):
    self.active = True
    self._agent_eps += 1
    self.step_rew = 0


  def after_agent_reset(self):
    self.active = False


  def after_env_step(self, obs, reward, done, info):
    self.ep_steps   += 1
    self.ep_reward  += reward
    self.step_rew   += reward
    # Prevent corrupting data if env.step() is incorrectly called when done=True
    self.env_done   = done if self.env_done is not None else None

    self._finish_episode()


  def _finish_episode(self):
    """
    This function can possibly be called on env.reset() instead of on env.step(). Both can possibly
    introduce problematic behavior:
      - env.reset(): If the last ever agent step corresponds to episode termination, env.reset()
        might never be called by the agent and the episode data will be lost
      - env.step(): Some env wrapper might accidentally lead to episode termination, without
        any intervention of the agent. For example, the Noop wrapper in Atari
    In both cases, for correct behavior, we need to track whether env_done == True. Otherwise, the
    episode data is not complete and MUST be ignored because it will corrupt the statistics.

    `env.step()` is chosen for the final implemntation, because
      - The problem with env.step() mentioned above is actually much less likely. Further, even if
        a wrapper leads to episode termination, the agent cannot do anything about this and data is
        still statistically correct
      - The problem with env.reset() is dependent on the agent implementation and might lead
        to more unstable behavior

    Note additionally, that calls to env.reset() by the agent do not necessarily mean that the
    environment is truly done.
    """

    # Ignore at initial reset and when env.step() is incorrectly called after done=True
    if self.env_done is None:
      return

    # Append episode data only if the env has advertised done
    if self.env_done:
      self.ep_lens.append(self.ep_steps)
      self.ep_rews.append(self.ep_reward)
      self._env_steps += self.ep_steps
      self.env_done = None


  def env_reset(self):
    self.ep_steps   = 0
    self.ep_reward  = 0
    self.env_done   = False
    self._env_eps   += 1


  def _init_stats(self):
    self.stats = {
      "mean_ep_rew":    -np.inf,
      "std_ep_rew":     -np.inf,
      "ep_len_mean":    -np.inf,
      "ep_len_std":     -np.inf,
      "best_mean_rew":  -np.inf,
      "best_ep_rew":    -np.inf,
      "last_log_ep":    0,
    }

    if self.mode == 't':
      self.stats["steps_per_sec"] = None          # Mean steps per second
      self.stats["last_log_time"] = time.time()   # Time at which the last runtime log happened
      self.stats["last_log_step"] = 0             # Step at which the last runtime log happened

    else:
      self.stats["score_episodes"] = 0


  def _init_stdout(self):

    n_eps = " (%d eps)"%(self.n_episodes) if self.n_episodes is not None else ""

    if self.mode == "t":
      # Function to compute the total time
      start_time = time.time()
      def total_time(_):
        delta = time.time() - start_time
        return '{0:02.0f}:{1:02.0f}'.format(*divmod(delta//60, 60))

      self.log_spec = [
        ("train/total_time",                      "",     total_time),
        ("train/mean_steps_per_sec",              ".3f",  lambda t: self.stats["steps_per_sec"]),
        ("train/agent_steps",                     "d",    lambda t: self._log_step),
        ("train/env_steps",                       "d",    lambda t: self._env_steps+self.ep_steps),
        ("train/episodes",                        "d",    lambda t: self._env_eps),
        ("train/best_episode_rew",                ".3f",  lambda t: self.stats["best_ep_rew"]),
        ("train/best_mean_ep_rew%s"%n_eps,        ".3f",  lambda t: self.stats["best_mean_rew"]),
        ("train/ep_len_mean%s"%n_eps,             ".3f",  lambda t: self.stats["ep_len_mean"]),
        ("train/ep_len_std %s"%n_eps,             ".3f",  lambda t: self.stats["ep_len_std"]),
        ("train/mean_ep_reward%s"%n_eps,          ".3f",  lambda t: self.stats["mean_ep_rew"]),
        ("train/std_ep_reward%s"%n_eps,           ".3f",  lambda t: self.stats["std_ep_rew"]),
      ]

    else:

      self.log_spec = [
        ("eval/agent_steps",                      "d",    lambda t: self._log_step),
        ("eval/best_episode_rew",                 ".3f",  lambda t: self.stats["best_ep_rew"]),
        ("eval/ep_len_mean",                      ".3f",  lambda t: self.stats["ep_len_mean"]),
        ("eval/ep_len_std ",                      ".3f",  lambda t: self.stats["ep_len_std"]),
        ("eval/mean_ep_reward",                   ".3f",  lambda t: self.stats["mean_ep_rew"]),
        ("eval/std_ep_reward",                    ".3f",  lambda t: self.stats["std_ep_rew"]),
        ("eval/score_episodes",                   "d",    lambda t: self.stats["score_episodes"]),
        ("eval/best_score",                       ".3f",  lambda t: self.stats["best_mean_rew"]),
      ]


  def set_stdout_logs(self, stdout_spec):
    """Build a list of tuples `(name, modifier, lambda)`. This list is
    used to print the runtime statistics logs to stdout. The tuple is defined as:
      `name`: `str`, the name of the reported value
      `modifier`: `str`, the type modifier for printing the value, e.g. `d` for int
      `lambda`: A function that takes the current **agent** timestep as argument
        and returns the value to be printed.
    Args:
      stdout_spec: `list`. Must have the same structure as the above list. Allows
        for logging custom data coming from the agent
    """
    if self.mode == 'e':
      return

    # Correct stdout_info
    filter_spec = []
    for log in stdout_spec:
      assert len(log) == 3
      assert callable(log[-1])
      assert not log[0].startswith("eval/")
      if not log[0].startswith("train/"):
        name = "train/" + log[0]
        log = (name,) + log[1:]
      filter_spec.append(log)
    stdout_spec = filter_spec

    # Update the stdout data
    self.log_spec = self.log_spec + stdout_spec


  def _build_stdout(self):
    """Build self.log_spec table"""
    if self.stdout_set:
      return
    self.stdout_set = True

    # Configure the TensorBoard stdout entries
    # Note that evaluation monitors don't output any TB data to stdout
    if self.mode == 't':

      # Get all summary Ops in the graph and their names
      summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
      names = []
      # Select the tag names which start with "stdout/" and correspond to scalar summaries
      for tensor in summary_ops:
        if not tensor.name.startswith("stdout/"):
          continue
        else:
          name = tensor.name[:tensor.name.rfind(":")]
          if tensor.op.node_def.op != "ScalarSummary":
            logger.warning("Cannot output non-scalar summary field %s to stdout", name)
            continue
          else:
            names.append(name)

      # Build the list of tuples for printing TB summary data to stdout
      # Make names appear with "debug/" prefix on stdout
      names   = sorted(names)
      tb_spec = [("debug" + name[6:], ".4f", lambda t, k=name: self.summary_dict[k]) for name in names]

      # Initialize the summary dictionary
      self.summary_dict = {name: np.nan for name in names}

    else:
      tb_spec = []

    # Do not include the TensorBoard data in stdout if no way to get the summary
    if len(tb_spec) > 0 and self._get_summary == self._default_summary_getter:
      logger.warning(
        "Found TensorFlow summary ops that need to be added to stdout, but cannot fetch "
        "the result of running the ops in a tf.Session(). Call %s to provide a method for "
        "fetching the summary data", self.__class__.__name__ + ".set_summary_getter()"
      )
    else:
      self.log_spec = self.log_spec + tb_spec

    # Format the stdout tuples
    self.log_spec = rltf_log.format_tabular(self.log_spec, sort=False)

    # Hacky - write the graph here because now we are sure the graph is ready
    # Write the graph to TensorBoard
    if self.mode == 't':
      graph = tf.get_default_graph()
      if len(graph.get_operations()) == 0:
        return
      graph_writer = tf.summary.FileWriter(self.tb_dir, filename_suffix=".graph")
      graph_writer.add_graph(graph)
      graph_writer.flush()
      graph_writer.close()



  def set_summary_getter(self, f):
    """Set a function that returns the TensorBoard summary to be saved to disk.
    The summaries need to be updated only at each logging period. The function must
    return an instance of tf.Summary or None, if no summary is available"""
    if self.mode == 't':
      self._get_summary = f
    else:
      logger.warning("Evaluation monitor does not support custom summary getters.")


  def _default_summary_getter(self):
    return tf.Summary()


  def _update_summary(self):
    # Get the most recent summary
    summary = self._get_summary()

    if self.mode == 't':
      summary = self._filter_summary(summary)

    self.summary = summary


  def _filter_summary(self, summary):
    # Do NOT filter if nothing from the summary needs to be redirected to stdout
    if len(self.summary_dict) == 0 or summary is None:
      return summary

    inds = []

    # Now remove all summary tags which start with "stdout/" and instead redirect them to stdout.
    # NOTE: This will ignore all manually added TB summary entries which start with "stdout/" and do
    # not actually exist as summary Ops in the TensorFlow graph. This is the expected behavior and
    # one should instead use set_stdout_logs(), since manually added data is most likely not dependent
    # on the graph and can be computed with a simple function, which is independent of sess.run()

    # Update the most recent values that need to be printed to stdout
    for i, v in enumerate(summary.value):
      if v.tag.startswith("stdout/"):
        if v.tag in self.summary_dict:
          self.summary_dict[v.tag] = v.simple_value
        # Make sure to remove all "stdout/" summary entries
        inds.append(i)

    # Remove the stdout summary entries so they don't clutter TB
    for i, j in enumerate(inds):
      summary.value.pop(j-i)

    return summary


  def _update_stats(self, info):
    """Update the values of the runtime statistics variables"""

    # Configure the stdout on the first logging event
    if not self.stdout_set:
      self._build_stdout()

    # Fetch and update the TensorBoard summary
    self._update_summary()

    stats = self.stats

    # Update train mode statistics
    if self.mode == 't':
      time_now      = time.time()
      steps_per_sec = (self._agent_steps - stats["last_log_step"]) / (time_now - stats["last_log_time"])
      lo            = stats["last_log_ep"] if self.n_episodes is None else -self.n_episodes

      stats["mean_ep_rew"]    = stats_mean(self.ep_rews[lo:])
      stats["std_ep_rew"]     = stats_std(self.ep_rews[lo:])
      stats["ep_len_mean"]    = stats_mean(self.ep_lens[lo:])
      stats["ep_len_std"]     = stats_std(self.ep_lens[lo:])
      stats["best_mean_rew"]  = max(stats["best_mean_rew"], stats["mean_ep_rew"])
      stats["best_ep_rew"]    = stats_best(stats["best_ep_rew"], self.ep_rews, stats["last_log_ep"])

      stats["last_log_ep"]    = len(self.ep_rews)
      stats["steps_per_sec"]  = steps_per_sec
      stats["last_log_time"]  = time_now
      stats["last_log_step"]  = self._agent_steps

    # Update eval mode statistics
    else:
      # Logging means that an evaluation run is finished, so it is time to update the statistics
      # In evaluation mode, statistics are always based on the most recent run

      last_log_ep   = self.stats_inds[-1] if len(self.stats_inds) > 0 else 0
      episode_rews  = self.ep_rews[last_log_ep:]
      episode_lens  = self.ep_lens[last_log_ep:]

      stats["mean_ep_rew"]    = stats_mean(episode_rews)
      stats["std_ep_rew"]     = stats_std(episode_rews)
      stats["ep_len_mean"]    = stats_mean(episode_lens)
      stats["ep_len_std"]     = stats_std(episode_lens)
      stats["best_mean_rew"]  = max(stats["best_mean_rew"], stats["mean_ep_rew"])
      stats["best_ep_rew"]    = stats_best(stats["best_ep_rew"], self.ep_rews, stats["last_log_ep"])
      stats["score_episodes"] = len(episode_rews)
      stats["last_log_ep"]    = len(self.ep_rews)

      # Append info with best_agent
      info["rltfmon.best_agent"] = stats["mean_ep_rew"] == stats["best_mean_rew"]

    # Update the stats logging steps and indices
    self.stats_steps.append(self._log_step)
    self.stats_inds.append(len(self.ep_rews))

    # Append the TensorBoard summary data
    if self.summary is not None:
      if self.mode == 't':
        self.summary.value.add(tag="train/mean_ep_rew", simple_value=stats["mean_ep_rew"])
      else:
        self.summary.value.add(tag="eval/mean_ep_rew",  simple_value=stats["mean_ep_rew"])


  def log_stats(self, info=None):
    """Log the training progress if the **agent** step is appropriate
    Args:
      info: The info dict returned after env.step(). Can be optionally appended with some data
    """

    if self._agent_steps % self.log_period != 0:
      return

    # Update the statistics
    self._update_stats(info)

    # Log the data to stdout
    stats_logger.info("")
    if self.mode == 't':
      for s, lambda_v in self.log_spec:
        stats_logger.info(s.format(lambda_v(self._agent_steps)))
    else:
      for s, lambda_v in self.log_spec:
        stats_logger.info(rltf_log.colorize(s.format(lambda_v(self._agent_steps)), "yellow"))
    stats_logger.info("")

    # Log the summary to TensorBoard
    if self.summary is not None:
      self.tb_writer.add_summary(self.summary, global_step=self._log_step)


  def save(self):
    """Save the statistics data to disk. Must be manually called"""

    if len(self.ep_rews) == 0:
      return

    data = {
      "env_steps":      self._env_steps,
      "agent_steps":    self._agent_steps,
      "env_episodes":   self._env_eps,
      "agent_episodes": self._agent_eps,
      "best_mean_rew":  self.stats["best_mean_rew"],
    }

    if self.autolog:
      data["log_period"] = self.log_period

    # Get the filenames
    if self.mode == 't':
      json_file         = "train_stats_summary.json"
      ep_rews_file      = "train_ep_rews.npy"
      ep_lens_file      = "train_ep_lens.npy"
      stats_inds_file   = "train_log_inds.npy"
      stats_steps_file  = "train_log_steps.npy"

    else:
      json_file         = "eval_stats_summary.json"
      ep_rews_file      = "eval_ep_rews.npy"
      ep_lens_file      = "eval_ep_lens.npy"
      stats_inds_file   = "eval_scores_inds.npy"
      stats_steps_file  = "eval_scores_steps.npy"

    # Write the data
    self._write_json(json_file, data)

    self._write_npy(ep_rews_file,     np.asarray(self.ep_rews, dtype=np.float32))
    self._write_npy(ep_lens_file,     np.asarray(self.ep_lens, dtype=np.int32))
    self._write_npy(stats_inds_file,  np.asarray(self.stats_inds, dtype=np.int32))
    self._write_npy(stats_steps_file, np.asarray(self.stats_steps, dtype=np.int32))

    # Flush the TB writer
    self.tb_writer.flush()


  def _resume(self):

    # Get the filenames
    if self.mode == 't':
      json_file         = "train_stats_summary.json"
      ep_rews_file      = "train_ep_rews.npy"
      ep_lens_file      = "train_ep_lens.npy"
      stats_inds_file   = "train_log_inds.npy"
      stats_steps_file  = "train_log_steps.npy"

    else:
      json_file         = "eval_stats_summary.json"
      ep_rews_file      = "eval_ep_rews.npy"
      ep_lens_file      = "eval_ep_lens.npy"
      stats_inds_file   = "eval_scores_inds.npy"
      stats_steps_file  = "eval_scores_steps.npy"

    # Read the JSON data
    data = self._read_json(json_file)

    if data:
      self._env_steps   = data["env_steps"]
      self._agent_steps = data["agent_steps"]
      self._env_eps     = data["env_episodes"]
      self._agent_eps   = data["agent_episodes"]
      self.stats["best_mean_rew"] = data["best_mean_rew"]

    # Read the numpy data
    self.ep_rews      = self._read_npy(ep_rews_file)
    self.ep_lens      = self._read_npy(ep_lens_file)
    self.stats_inds   = self._read_npy(stats_inds_file)
    self.stats_steps  = self._read_npy(stats_steps_file)


  def close(self):
    self.tb_writer.close()


  def _read_npy(self, file):
    file = os.path.join(self.log_dir, file)
    if os.path.exists(file):
      return list(np.load(file))
    return []


  def _write_npy(self, file, data):
    file = os.path.join(self.log_dir, file)
    with atomic_write.atomic_write(file, True) as f:
      np.save(f, data)


  def _read_json(self, file):
    file = os.path.join(self.log_dir, file)
    if not os.path.exists(file):
      return {}
    with open(file, 'r') as f:
      data = json.load(f)
    return data


  def _write_json(self, file, data):
    file = os.path.join(self.log_dir, file)
    with atomic_write.atomic_write(file) as f:
      json.dump(data, f, indent=4, sort_keys=True)


  @property
  def _log_step(self):
    if self._mode == 't':
      return self._agent_steps
    else:
      # Scale evaluation mode steps to agent training steps
      return int(self._agent_steps // self.log_period * self.eval_period)


  @property
  def mode(self):
    return self._mode

  @property
  def agent_steps(self):
    return self._agent_steps

  @property
  def env_steps(self):
    return self._env_steps

  @property
  def agent_eps(self):
    return self._agent_eps

  @property
  def env_eps(self):
    return self._env_eps

  @property
  def episode_rews(self):
    return list(self.ep_rews)

  @property
  def episode_lens(self):
    return list(self.ep_lens)


def stats_mean(data):
  if len(data) > 0:
    return np.mean(data)
  return -np.inf


def stats_std(data):
  if len(data) > 0:
    return np.std(data)
  return np.nan


def stats_best(best, data, i):
  if len(data[i:]) > 0:
    data_best = np.max(data[i:])
    return max(best, data_best)
  return best
