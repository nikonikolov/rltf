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


class StatsRecorder:

  def __init__(self, log_dir, mode='t', n_ep_stats=100):
    """
    Args:
      log_dir: str. The path for the directory where the videos are saved
      n_ep_stats: int. Number of episodes over which to report the runtime statistitcs
    """

    # Member data
    self.log_dir    = log_dir
    self.n_ep_stats = n_ep_stats
    self.log_info   = None

    # NOTE: self.ep_id is the episode ID as seen by the agent and is not guaranteed to be the same as
    # the length of ep_lens. For example, in Atari environments end of life = end of episode as seen
    # by the agent but not by the environment

    # Statistics
    self.ep_rews = []
    self.ep_lens = []
    self.stats   = None   # A dictionary with runtime statistics
    self.ep_id   = 0

    # Runtime variables
    self.ep_reward  = None
    self.ep_steps   = None
    self.env_steps  = 0         # Total number of environment steps
    self._mode      = mode      # Running mode: either 't' (train) or 'e' (eval)
    self.done       = None

    # Members specific to train mode
    self.steps_p_s      = None
    self.t_last_log     = time.time()   # Time at which the last runtime log happened
    self.step_last_log  = 0             # Step at which the last runtime log happened

    # Members specific to evaluation mode
    self._in_eval_run       = False
    self.eval_run_rews      = None  # Episode rewards of the current eval run
    self.eval_run_lens      = None  # Episode lengths of the current eval run
    self.eval_scores_steps  = []    # The final agent step of each evaluation score
    self.eval_scores_inds   = []    # Indices for the final episode in each eval run (exclusive index)

    self._init_stats()

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
      "ep_last_stats":  0,
    }

    if self._mode == 'e':
      self.stats["best_score"]     = -np.inf
      self.stats["score_mean"]     = -np.inf
      self.stats["score_std"]      = -np.nan
      self.stats["score_episodes"] = 0


  def after_step(self, obs, reward, done, info):
    self.ep_steps   += 1
    self.ep_reward  += reward
    self.done       = done


  def _finish_episode(self):
    """This function can be called either on env.reset() or on env.step() when done=True. The former
    is more clear and thus chosen as the final implementation. Either way, if self.done = False,
    the episode data MUST be ignored. Otherwise, the data will corrupt the statistics. The most likely
    case for calling reset() in the middle of the episode is that evaluation run has reached max steps
    without being able to complete the episode.
    """
    # Ignore at initial reset
    if self.done is None:
      return

    self.env_steps += self.ep_steps

    if self.done:
      if self._mode == 't':
        self.ep_lens.append(self.ep_steps)
        self.ep_rews.append(self.ep_reward)
      else:
        self.eval_run_lens.append(self.ep_steps)
        self.eval_run_rews.append(self.ep_reward)


  def reset(self):
    self._finish_episode()

    self.ep_steps   = 0
    self.ep_reward  = 0
    self.done       = False
    self.ep_id      += 1


  def define_log_info(self, custom_log_info):
    """Build a list of tuples `(name, modifier, lambda)`. This list is
    used to print the runtime statistics logs. The tuple is defined as:
      `name`: `str`, the name of the reported value
      `modifier`: `str`, the type modifier for printing the value, e.g. `d` for int
      `lambda`: A function that takes the current **agent** timestep as argument
        and returns the value to be printed.
    Args:
      custom_log_info: `list`. Must have the same structure as the above list. Allows
        for logging custom data coming from the agent
    """

    n = self.n_ep_stats

    if self._mode == "t":
      log_info = [
        ("train/mean_steps_per_sec",              ".3f",  lambda t: self.steps_p_s),

        ("train/agent_steps",                     "d",    lambda t: t),
        ("train/env_steps",                       "d",    lambda t: self.env_steps+self.ep_steps),
        ("train/episodes",                        "d",    lambda t: self.ep_id),
        ("train/best_episode_rew",                ".3f",  lambda t: self.stats["best_ep_rew"]),
        ("train/best_mean_ep_rew (%d eps)"%n,     ".3f",  lambda t: self.stats["best_mean_rew"]),
        ("train/ep_len_mean (%d eps)"%n,          ".3f",  lambda t: self.stats["ep_len_mean"]),
        ("train/ep_len_std  (%d eps)"%n,          ".3f",  lambda t: self.stats["ep_len_std"]),
        ("train/mean_ep_reward (%d eps)"%n,       ".3f",  lambda t: self.stats["mean_ep_rew"]),
        ("train/std_ep_reward (%d eps)"%n,        ".3f",  lambda t: self.stats["std_ep_rew"]),
      ]

      log_info = log_info + custom_log_info

    else:
      log_info = [
        ("eval/agent_steps",                      "d",    lambda t: t),
        ("eval/env_steps",                        "d",    lambda t: self.env_steps+self.ep_steps),
        ("eval/episodes",                         "d",    lambda t: self.ep_id),
        ("eval/best_episode_rew",                 ".3f",  lambda t: self.stats["best_ep_rew"]),
        ("eval/best_mean_ep_rew (%d eps)"%n,      ".3f",  lambda t: self.stats["best_mean_rew"]),
        ("eval/ep_len_mean (%d eps)"%n,           ".3f",  lambda t: self.stats["ep_len_mean"]),
        ("eval/ep_len_std  (%d eps)"%n,           ".3f",  lambda t: self.stats["ep_len_std"]),
        ("eval/mean_ep_reward (%d eps)"%n,        ".3f",  lambda t: self.stats["mean_ep_rew"]),
        ("eval/std_ep_reward (%d eps)"%n,         ".3f",  lambda t: self.stats["std_ep_rew"]),
        ("eval/best_score",                       ".3f",  lambda t: self.stats["best_score"]),
        ("eval/score_episodes",                   "d",    lambda t: self.stats["score_episodes"]),
        ("eval/score_mean",                       ".3f",  lambda t: self.stats["score_mean"]),
        ("eval/score_std",                        ".3f",  lambda t: self.stats["score_std"]),
      ]

    self.log_info = rltf_log.format_tabular(log_info, sort=False)


  def _stats_mean(self, data):
    if len(data) > 0:
      return np.mean(data[-self.n_ep_stats:])
    return np.nan


  def _stats_std(self, data):
    if len(data) > 0:
      return np.std(data[-self.n_ep_stats:])
    return np.nan


  def _compute_runtime_stats(self, step):
    """Update the values of the runtime statistics variables"""

    def _stats_max(data, i=0):
      if len(data[i:]) > 0:
        return np.max(data[i:])
      return -np.inf

    self.stats["mean_ep_rew"]   = self._stats_mean(self.ep_rews)
    self.stats["std_ep_rew"]    = self._stats_std(self.ep_rews)
    self.stats["ep_len_mean"]   = self._stats_mean(self.ep_lens)
    self.stats["ep_len_std"]    = self._stats_std(self.ep_lens)
    self.stats["best_mean_rew"] = max(self.stats["best_mean_rew"], self.stats["mean_ep_rew"])
    best_ep_rew = _stats_max(self.ep_rews, self.stats["ep_last_stats"])
    self.stats["best_ep_rew"] = max(self.stats["best_ep_rew"], best_ep_rew)
    self.stats["ep_last_stats"] = len(self.ep_rews)

    if self._mode == 't':
      t_now               = time.time()
      self.steps_p_s      = (step - self.step_last_log) / (t_now - self.t_last_log)
      self.step_last_log  = step
      self.t_last_log     = t_now


  def log_stats(self, t):
    """Log the training progress
    Args:
      t: int. Current **agent** timestep
    """

    # Update the statistics
    self._compute_runtime_stats(t)

    stats_logger.info("")
    for s, lambda_v in self.log_info:
      stats_logger.info(s.format(lambda_v(t)))
    stats_logger.info("")


  def save(self):
    """Save the statistics data to disk. Must be manually called"""

    if len(self.ep_rews) == 0:
      return

    data = {
      "env_steps":      self.env_steps,
      "episodes":       self.ep_id,
      "best_mean_rew":  self.stats["best_mean_rew"],
    }

    if self._mode == 't':
      json_file     = "train_stats_summary.json"
      ep_rews_file  = "train_ep_rews.npy"
      ep_lens_file  = "train_ep_lens.npy"

    else:
      json_file     = "eval_stats_summary.json"
      ep_rews_file  = "eval_ep_rews.npy"
      ep_lens_file  = "eval_ep_lens.npy"

      data["best_score"] = self.stats["best_score"],

      self._write_npy("eval_scores_inds.npy",  np.asarray(self.eval_scores_inds, dtype=np.int32))
      self._write_npy("eval_scores_steps.npy", np.asarray(self.eval_scores_steps, dtype=np.int32))

    self._write_json(json_file, data)

    self._write_npy(ep_rews_file, np.asarray(self.ep_rews, dtype=np.float32))
    self._write_npy(ep_lens_file, np.asarray(self.ep_lens, dtype=np.int32))


  def _resume(self):

    if self._mode == 't':
      json_file     = "train_stats_summary.json"
      ep_rews_file  = "train_ep_rews.npy"
      ep_lens_file  = "train_ep_lens.npy"

    else:
      json_file     = "eval_stats_summary.json"
      ep_rews_file  = "eval_ep_rews.npy"
      ep_lens_file  = "eval_ep_lens.npy"

      self.eval_scores_inds   = self._read_npy("eval_scores_inds.npy")
      self.eval_scores_steps  = self._read_npy("eval_scores_steps.npy")

    data = self._read_json(json_file)

    if data:
      self.env_steps  = data["env_steps"]
      self.ep_id      = data["episodes"]
      self.stats["best_mean_rew"] = data["best_mean_rew"]

      if self._mode == 'e':
        self.stats["best_score"] = data["best_score"]

    self.ep_rews = self._read_npy(ep_rews_file)
    self.ep_lens = self._read_npy(ep_lens_file)


  @property
  def mode(self):
    return self._mode


  @property
  def episode_id(self):
    return self.ep_id


  @property
  def total_steps(self):
    return self.env_steps + self.ep_steps


  @property
  def mean_ep_rew(self):
    return self._stats_mean(self.ep_rews)


  @property
  def eval_score(self):
    assert self._mode == 'e'
    return self.stats["score_mean"]


  @property
  def episode_rewards(self):
    return list(self.ep_rews)


  @property
  def episode_lens(self):
    return list(self.ep_lens)


  def start_eval_run(self):
    """Signal that eval run begins. Resets the eval temporary data"""
    assert not self._in_eval_run
    self._in_eval_run = True

    self.eval_run_rews = []
    self.eval_run_lens = []

    self.stats["score_mean"]     = -np.inf
    self.stats["score_std"]      = np.nan
    self.stats["score_episodes"] = 0


  def end_eval_run(self, t):
    """Signal the end of the eval run
    Args:
      t: int. Current agent step
    Returns:
      True if agent is best so far, False otherwise
    """
    assert self._in_eval_run
    self._in_eval_run = False

    episodes = len(self.eval_run_rews)

    if episodes > 0:
      score_mean = np.mean(self.eval_run_rews)
      score_std  = np.std(self.eval_run_rews)
    else:
      score_mean = -np.inf
      score_std  = np.nan

    self.stats["score_mean"]     = score_mean
    self.stats["score_std"]      = score_std
    self.stats["score_episodes"] = episodes

    if score_mean > self.stats["best_score"]:
      self.stats["best_score"] = score_mean
      best_agent = True
    else:
      best_agent = False

    # Append the episode statistics
    self.ep_rews += self.eval_run_rews
    self.ep_lens += self.eval_run_lens

    # Append the evaluation score data
    self.eval_scores_steps.append(t)
    self.eval_scores_inds.append(len(self.ep_rews))

    return best_agent


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
