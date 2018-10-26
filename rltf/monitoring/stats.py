import json
import logging
import os
import time
import numpy as np

# from gym        import error
from gym.utils  import atomic_write

from rltf.utils import rltf_log
import rltf.conf

logger        = logging.getLogger(__name__)
stats_logger  = logging.getLogger(rltf.conf.STATS_LOGGER_NAME)

# Constant for number of most recent episodes over which to report some of the runtime statistitcs
N_EPS_STATS   = 100


class StatsRecorder:

  def __init__(self, log_dir, log_period, mode):
    """
    Args:
      log_dir: str. The path for the directory where the videos are saved
      log_period: int. The period for logging statistics. If mode == 'e', then it must equal
        the evaluation length in order to keep correct evaluation statistics
      mode: str. Either 't' (for train) or `e` (for eval)
    """

    # Member data
    self.log_dir    = log_dir
    self.log_period = log_period
    self.log_info   = None
    self._mode      = mode      # Running mode: either 't' (train) or 'e' (eval)

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

    # Episode tracking
    self.ep_reward  = None      # Track the episode reward so far
    self.step_rew   = None      # Track the total reward from the current agent step
    self.ep_steps   = None      # Track the episode environment steps so far
    self.env_done   = None      # Track whether the environment is done

    # Initialize self.stats
    self._init_stats()

    # Create log_dir or resume with saved data if log_dir exists
    if not os.path.exists(self.log_dir):
      logger.info('Creating stats directory %s', self.log_dir)
      os.makedirs(self.log_dir)
    elif len(os.listdir(self.log_dir)) > 0:
      logger.info('Resuming with stats data from %s', self.log_dir)
      self._resume()


  def _init_stats(self):
    self.stats = {
      "mean_ep_rew":    np.nan,
      "std_ep_rew":     np.nan,
      "ep_len_mean":    np.nan,
      "ep_len_std":     np.nan,
      "best_mean_rew":  -np.inf,
      "best_ep_rew":    -np.inf,
      "last_log_ep":  0,
    }

    if self.mode == 't':
      self.stats["steps_per_sec"] = None          # Mean steps per second
      self.stats["last_log_time"] = time.time()   # Time at which the last runtime log happened
      self.stats["last_log_step"] = 0             # Step at which the last runtime log happened

    else:
      self.stats["best_score"]     = -np.inf
      self.stats["score_mean"]     = -np.inf
      self.stats["score_std"]      = -np.nan
      self.stats["score_episodes"] = 0


  def before_agent_step(self, action):
    self._agent_steps += 1
    # Keep this here. Sometimes, env.reset() might induce env.step(), but not
    # agent.env.step() due to Wrapper behavior
    self.step_rew = 0


  def after_agent_step(self, obs, reward, done, info):
    # Append stats data for the agent
    if "rltf_mon" in info:
      data = info["rltf_mon"]
    else:
      data = {}
      info["rltf_mon"] = data
    data["ep_rew"]   = self.ep_reward
    data["step_rew"] = self.step_rew

    # NOTE:
    # - Logging needs to happen here, NOT in after_env_step. The latter might see the same value of
    #   self._agent_steps in more than one call and output several logging messages with the same data
    # - Logging TensorBoard summary data here does not interfere with any agent training step, even
    #   if it is concurrently executed, as long as summary has been updated sufficiently recently
    self.log_stats(data)


  def agent_reset(self):
    self._agent_eps += 1
    self.step_rew = 0


  def after_env_step(self, obs, reward, done, info):
    self.ep_steps   += 1
    self.ep_reward  += reward
    self.step_rew   += reward
    self.env_done   = done

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

    # Ignore at initial reset
    if self.env_done is None:
      return

    self._env_steps += self.ep_steps

    # Append episode data only if the env has advertised done
    if self.env_done:
      self.ep_lens.append(self.ep_steps)
      self.ep_rews.append(self.ep_reward)


  def env_reset(self):
    self.ep_steps   = 0
    self.ep_reward  = 0
    self.env_done   = False
    self._env_eps   += 1


  def set_stdout_logs(self, custom_log_info):
    """Build a list of tuples `(name, modifier, lambda)`. This list is
    used to print the runtime statistics logs to stdout. The tuple is defined as:
      `name`: `str`, the name of the reported value
      `modifier`: `str`, the type modifier for printing the value, e.g. `d` for int
      `lambda`: A function that takes the current **agent** timestep as argument
        and returns the value to be printed.
    Args:
      custom_log_info: `list`. Must have the same structure as the above list. Allows
        for logging custom data coming from the agent
    """

    # Correct custom_log_info
    correct_info = []
    for log in custom_log_info:
      assert len(log) == 3
      assert callable(log[-1])
      assert not log[0].startswith("eval/")
      if not log[0].startswith("train/"):
        name = "train/" + log[0]
        log = (name,) + log[1:]
      correct_info.append(log)
    custom_log_info = correct_info

    N = N_EPS_STATS

    if self.mode == "t":
      log_info = [
        ("train/mean_steps_per_sec",              ".3f",  lambda t: self.stats["steps_per_sec"]),

        ("train/agent_steps",                     "d",    lambda t: t),
        ("train/env_steps",                       "d",    lambda t: self._env_steps+self.ep_steps),
        ("train/episodes",                        "d",    lambda t: self._env_eps),
        ("train/best_episode_rew",                ".3f",  lambda t: self.stats["best_ep_rew"]),
        ("train/best_mean_ep_rew (%d eps)"%N,     ".3f",  lambda t: self.stats["best_mean_rew"]),
        ("train/ep_len_mean (%d eps)"%N,          ".3f",  lambda t: self.stats["ep_len_mean"]),
        ("train/ep_len_std  (%d eps)"%N,          ".3f",  lambda t: self.stats["ep_len_std"]),
        ("train/mean_ep_reward (%d eps)"%N,       ".3f",  lambda t: self.stats["mean_ep_rew"]),
        ("train/std_ep_reward (%d eps)"%N,        ".3f",  lambda t: self.stats["std_ep_rew"]),
      ]

      log_info = log_info + custom_log_info

    else:
      log_info = [
        ("eval/agent_steps",                      "d",    lambda t: t),
        ("eval/env_steps",                        "d",    lambda t: self._env_steps+self.ep_steps),
        ("eval/episodes",                         "d",    lambda t: self._env_eps),
        ("eval/best_episode_rew",                 ".3f",  lambda t: self.stats["best_ep_rew"]),
        ("eval/best_mean_ep_rew (%d eps)"%N,      ".3f",  lambda t: self.stats["best_mean_rew"]),
        ("eval/ep_len_mean (%d eps)"%N,           ".3f",  lambda t: self.stats["ep_len_mean"]),
        ("eval/ep_len_std  (%d eps)"%N,           ".3f",  lambda t: self.stats["ep_len_std"]),
        ("eval/mean_ep_reward (%d eps)"%N,        ".3f",  lambda t: self.stats["mean_ep_rew"]),
        ("eval/std_ep_reward (%d eps)"%N,         ".3f",  lambda t: self.stats["std_ep_rew"]),
        ("eval/best_score",                       ".3f",  lambda t: self.stats["best_score"]),
        ("eval/score_episodes",                   "d",    lambda t: self.stats["score_episodes"]),
        ("eval/score_mean",                       ".3f",  lambda t: self.stats["score_mean"]),
        ("eval/score_std",                        ".3f",  lambda t: self.stats["score_std"]),
      ]

    self.log_info = rltf_log.format_tabular(log_info, sort=False)


  def _update_stats(self, info):
    """Update the values of the runtime statistics variables"""

    stats = self.stats

    # Update standard statisticis
    stats["mean_ep_rew"]    = stats_mean(self.ep_rews)
    stats["std_ep_rew"]     = stats_std(self.ep_rews)
    stats["ep_len_mean"]    = stats_mean(self.ep_lens)
    stats["ep_len_std"]     = stats_std(self.ep_lens)
    stats["best_mean_rew"]  = max(stats["best_mean_rew"], stats["mean_ep_rew"])
    stats["best_ep_rew"]    = stats_best(stats["best_ep_rew"], self.ep_rews, stats["last_log_ep"])
    stats["last_log_ep"]    = len(self.ep_rews)

    # Update statistics specific to train mode
    if self.mode == 't':
      time_now      = time.time()
      steps_per_sec = (self._agent_steps - stats["last_log_step"]) / (time_now - stats["last_log_time"])

      stats["steps_per_sec"] = steps_per_sec
      stats["last_log_time"] = time_now
      stats["last_log_step"] = self._agent_steps

    # Update statistics specific to eval mode
    else:
      # Logging means that an evaluation run is finished, so it is time to update the statistics
      # last_log_ep   = stats["last_log_ep"]
      last_log_ep   = self.stats_inds[-1] if len(self.stats_inds) > 0 else 0
      episode_data  = self.ep_rews[last_log_ep:]

      stats["score_mean"]     = stats_mean(episode_data)
      stats["score_std"]      = stats_std(episode_data)
      stats["score_episodes"] = len(episode_data)
      stats["best_score"]     = max(stats["score_mean"], stats["best_score"])

      # Append info with best_agent
      info["best_agent"] = stats["score_mean"] == stats["best_score"]

    # Update the stats logging steps and indices
    self.stats_steps.append(self._agent_steps)
    self.stats_inds.append(len(self.ep_rews))


  def log_stats(self, info):
    """Log the training progress if the **agent** step is appropriate
    Args:
      info: The info dict returned after env.step(). Can be optionally appended with some data
    """

    if self._agent_steps % self.log_period != 0:
      return

    # TODO: Add TB here

    # Update the statistics
    self._update_stats(info)

    # Log the data to stdout
    stats_logger.info("")
    if self.mode == 't':
      for s, lambda_v in self.log_info:
        stats_logger.info(s.format(lambda_v(self._agent_steps)))
    else:
      for s, lambda_v in self.log_info:
        stats_logger.info(rltf_log.colorize(s.format(lambda_v(self._agent_steps)), "yellow"))
    stats_logger.info("")


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

      data["best_score"] = self.stats["best_score"],

    # Write the data
    self._write_json(json_file, data)

    self._write_npy(ep_rews_file,     np.asarray(self.ep_rews, dtype=np.float32))
    self._write_npy(ep_lens_file,     np.asarray(self.ep_lens, dtype=np.int32))
    self._write_npy(stats_inds_file,  np.asarray(self.stats_inds, dtype=np.int32))
    self._write_npy(stats_steps_file, np.asarray(self.stats_steps, dtype=np.int32))


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

      if self.mode == 'e':
        self.stats["best_score"] = data["best_score"]

    # Read the numpy data
    self.ep_rews      = self._read_npy(ep_rews_file)
    self.ep_lens      = self._read_npy(ep_lens_file)
    self.stats_inds   = self._read_npy(stats_inds_file)
    self.stats_steps  = self._read_npy(stats_steps_file)


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
  def mean_ep_rew(self):
    return stats_mean(self.ep_rews)

  # TODO: Remove this - currently used in Agent log_stats
  @property
  def eval_score(self):
    assert self.mode == 'e'
    return self.stats["score_mean"]

  @property
  def episode_rews(self):
    return list(self.ep_rews)

  @property
  def episode_lens(self):
    return list(self.ep_lens)


def stats_mean(data):
  if len(data) > 0:
    return np.mean(data[-N_EPS_STATS:])
  return -np.inf


def stats_std(data):
  if len(data) > 0:
    return np.std(data[-N_EPS_STATS:])
  return np.nan


def stats_best(best, data, i):
  if len(data[i:]) > 0:
    data_best = np.max(data[i:])
    return max(best, data_best)
  return best

# def stats_max(data, i=0):
#   if len(data[i:]) > 0:
#     return np.max(data[i:])
#   return -np.inf
