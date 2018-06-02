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
    self.steps_p_s  = None
    self.t_last_log     = time.time()   # Time at which the last runtime log happened
    self.step_last_log  = 0             # Step at which the last runtime log happened

    # NOTE: self.train_ep_id and self.eval_ep_id are the episode IDs as seen by the agent.
    # The length of train_ep_lens and eval_ep_lens is not guaranteed to be the same. For
    # example, in Atari environments end of life = end of episode as seen by the agent but
    # not by the environment

    # Training statistics
    self.train_ep_rews = []
    self.train_ep_lens = []
    self.train_steps   = 0      # Total number of environment steps in train mode
    self.train_stats   = None   # A dictionary with runtime statistics about training
    self.train_ep_id   = 0

    # Evaluation statistics
    self.eval_ep_rews = []
    self.eval_ep_lens = []
    self.eval_ep_inds = []      # Indices for the final episode in each eval run (exclusive index)
    self.eval_steps   = 0       # Total number of environment steps in eval mode
    self.eval_stats   = None    # A dictionary with runtime statistics about evaluation
    self.eval_ep_id   = 0

    self.eval_run_rews      = None  # Episode rewards of the current eval run
    self.eval_run_lens      = None  # Episode lengths of the current eval run
    self.eval_scores_rews   = []    # Evaluation scores so far
    self.eval_scores_steps  = []    # The final agent step of each evaluation score

    # Runtime variables
    self.ep_reward  = None
    self.ep_steps   = None
    self.env_steps  = 0         # Total number of environment steps in any mode
    self._mode      = mode      # Running mode: either 't' (train) or 'e' (eval)
    self.done       = None
    self._in_eval_run = False

    self._init_stats()

    if not os.path.exists(self.log_dir):
      logger.info('Creating stats directory %s', self.log_dir)
      os.makedirs(self.log_dir)
    elif len(os.listdir(self.log_dir)) > 0:
      logger.info('Resuming with stats data from %s', self.log_dir)
      self._resume()


  def _init_stats(self):
    if self._mode == 't':
      self.train_stats = {
        "mean_ep_rew":    float("nan"),
        "std_ep_rew":     float("nan"),
        "ep_len_mean":    float("nan"),
        "ep_len_std":     float("nan"),
        "best_mean_rew":  -float("inf"),
        "best_ep_rew":    -float("inf"),
        "ep_last_stats":  0,
      }

    else:
      self.eval_stats = {
        "mean_ep_rew":    float("nan"),
        "std_ep_rew":     float("nan"),
        "ep_len_mean":    float("nan"),
        "ep_len_std":     float("nan"),
        "best_mean_rew":  -float("inf"),
        "best_ep_rew":    -float("inf"),
        "ep_last_stats":  0,
        "best_score":     -float("inf"),
        "score_mean":     -float("inf"),
        "score_std":      float("nan"),
        "score_episodes": 0,
      }


  def after_step(self, obs, reward, done, info):
    self.ep_steps   += 1
    self.env_steps  += 1
    self.ep_reward  += reward
    self.done       = done


  def _finish_episode(self):
    """This function can be called on reset() or on step() when done=True. The former is more clear.
    Either way, if self.done = False, the episode data MUST be ignored. Otherwise, the data
    will corrupt the statistics. The most likely case for calling reset() in the middle of the
    episode is that evaluation run has reached max steps without being able to complete the episode.
    """
    if self.done is None:
      return

    # The mode at the end of the episode is used to determine how to record the statistics
    if self._mode == 't':
      self.train_steps += self.ep_steps
      if self.done:
        self.train_ep_lens.append(self.ep_steps)
        self.train_ep_rews.append(self.ep_reward)
    else:
      self.eval_steps += self.ep_steps

      if self.done:
        self.eval_run_lens.append(self.ep_steps)
        self.eval_run_rews.append(self.ep_reward)


  def reset(self):
    self._finish_episode()

    self.ep_steps   = 0
    self.ep_reward  = 0
    self.done       = False

    if self._mode == 't':
      self.train_ep_id += 1
    else:
      self.eval_ep_id  += 1


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

    default_info = [
      ("train/mean_steps_per_sec",              ".3f",  lambda t: self.steps_p_s),

      ("train/agent_steps",                     "d",    lambda t: t),
      ("train/env_steps",                       "d",    lambda t: self.train_steps+self.ep_steps),
      ("train/episodes",                        "d",    lambda t: self.train_ep_id),
      ("train/best_episode_rew",                ".3f",  lambda t: self.train_stats["best_ep_rew"]),
      ("train/best_mean_ep_rew (%d eps)"%n,     ".3f",  lambda t: self.train_stats["best_mean_rew"]),
      ("train/ep_len_mean (%d eps)"%n,          ".3f",  lambda t: self.train_stats["ep_len_mean"]),
      ("train/ep_len_std  (%d eps)"%n,          ".3f",  lambda t: self.train_stats["ep_len_std"]),
      ("train/mean_ep_reward (%d eps)"%n,       ".3f",  lambda t: self.train_stats["mean_ep_rew"]),
      ("train/std_ep_reward (%d eps)"%n,        ".3f",  lambda t: self.train_stats["std_ep_rew"]),


      ("eval/agent_steps",                      "d",    lambda t: t),
      ("eval/env_steps",                        "d",    lambda t: self.eval_steps+self.ep_steps),
      ("eval/episodes",                         "d",    lambda t: self.eval_ep_id),
      ("eval/best_episode_rew",                 ".3f",  lambda t: self.eval_stats["best_ep_rew"]),
      ("eval/best_mean_ep_rew (%d eps)"%n,      ".3f",  lambda t: self.eval_stats["best_mean_rew"]),
      ("eval/ep_len_mean (%d eps)"%n,           ".3f",  lambda t: self.eval_stats["ep_len_mean"]),
      ("eval/ep_len_std  (%d eps)"%n,           ".3f",  lambda t: self.eval_stats["ep_len_std"]),
      ("eval/mean_ep_reward (%d eps)"%n,        ".3f",  lambda t: self.eval_stats["mean_ep_rew"]),
      ("eval/std_ep_reward (%d eps)"%n,         ".3f",  lambda t: self.eval_stats["std_ep_rew"]),
      ("eval/best_score",                       ".3f",  lambda t: self.eval_stats["best_score"]),
      ("eval/score_episodes",                   "d",    lambda t: self.eval_stats["score_episodes"]),
      ("eval/score_mean",                       ".3f",  lambda t: self.eval_stats["score_mean"]),
      ("eval/score_std",                        ".3f",  lambda t: self.eval_stats["score_std"]),
    ]

    log_info = default_info + custom_log_info
    self.log_info = rltf_log.format_tabular(log_info, sort=False)


  def _stats_mean(self, data):
    if len(data) > 0:
      return np.mean(data[-self.n_ep_stats:])
    return float("nan")


  def _stats_std(self, data):
    if len(data) > 0:
      return np.std(data[-self.n_ep_stats:])
    return float("nan")


  def _compute_runtime_stats(self, step):
    """Update the values of the runtime statistics variables"""

    def _stats_max(data, i=0):
      if len(data[i:]) > 0:
        return np.max(data[i:])
      return -float("inf")

    if self._mode == 't':
      self.train_stats["mean_ep_rew"] = self._stats_mean(self.train_ep_rews)
      self.train_stats["std_ep_rew"]  = self._stats_std(self.train_ep_rews)
      self.train_stats["ep_len_mean"] = self._stats_mean(self.train_ep_lens)
      self.train_stats["ep_len_std"]  = self._stats_std(self.train_ep_lens)
      self.train_stats["best_mean_rew"] = max(self.train_stats["best_mean_rew"],
                                              self.train_stats["mean_ep_rew"])
      best_ep_rew = _stats_max(self.train_ep_rews, self.train_stats["ep_last_stats"])
      self.train_stats["best_ep_rew"] = max(self.train_stats["best_ep_rew"], best_ep_rew)
      self.train_stats["ep_last_stats"] = len(self.train_ep_rews)

    else:
      self.eval_stats["mean_ep_rew"] = self._stats_mean(self.eval_ep_rews)
      self.eval_stats["std_ep_rew"]  = self._stats_std(self.eval_ep_rews)
      self.eval_stats["ep_len_mean"] = self._stats_mean(self.eval_ep_lens)
      self.eval_stats["ep_len_std"]  = self._stats_std(self.eval_ep_lens)
      self.eval_stats["best_mean_rew"] = max(self.eval_stats["best_mean_rew"],
                                             self.eval_stats["mean_ep_rew"])
      best_ep_rew = _stats_max(self.eval_ep_rews, self.eval_stats["ep_last_stats"])
      self.eval_stats["best_ep_rew"] = max(self.eval_stats["best_ep_rew"], best_ep_rew)
      self.eval_stats["ep_last_stats"] = len(self.eval_ep_rews)

    t_now  = time.time()
    if self._mode == 't':
      self.steps_p_s = (step - self.step_last_log) / (t_now - self.t_last_log)
      self.step_last_log = step
    self.t_last_log = t_now


  def log_stats(self, t):
    """Log the training progress
    Args:
      t: int. Current **agent** timestep
    """

    # Update the statistics
    self._compute_runtime_stats(t)

    stats_logger.info("")
    for s, lambda_v in self.log_info:
      if self._mode == 't' and not s.startswith("| eval/"):
        stats_logger.info(s.format(lambda_v(t)))
      elif self._mode == 'e' and not s.startswith("| train/"):
        stats_logger.info(s.format(lambda_v(t)))
    stats_logger.info("")


  def save(self):
    """Save the statistics data to disk. Must be manually called"""

    if self._mode == 't' and len(self.train_ep_rews) > 0:

      data = {
        "total_env_steps":  self.env_steps,
        "train_steps":      self.train_steps,
        "train_episodes":   self.train_ep_id,
        "best_mean_rew":    self.train_stats["best_mean_rew"],
      }
      self._write_json("train_stats_summary.json", data)

      self._write_npy("train_ep_rews.npy", np.asarray(self.train_ep_rews, dtype=np.float32))
      self._write_npy("train_ep_lens.npy", np.asarray(self.train_ep_lens, dtype=np.int32))


    if self._mode == 'e' and len(self.eval_ep_rews) > 0:

      data = {
        "total_env_steps":  self.env_steps,
        "eval_steps":       self.eval_steps,
        "eval_episodes":    self.eval_ep_id,
        "best_mean_rew":    self.eval_stats["best_mean_rew"],
        "best_score":       self.eval_stats["best_score"],
      }
      self._write_json("eval_stats_summary.json", data)

      self._write_npy("eval_ep_rews.npy", np.asarray(self.eval_ep_rews, dtype=np.float32))
      self._write_npy("eval_ep_lens.npy", np.asarray(self.eval_ep_lens, dtype=np.int32))
      self._write_npy("eval_ep_inds.npy", np.asarray(self.eval_ep_inds, dtype=np.int32))
      self._write_npy("eval_scores_rews.npy",  np.asarray(self.eval_scores_rews,  dtype=np.float32))
      self._write_npy("eval_scores_steps.npy", np.asarray(self.eval_scores_steps, dtype=np.int32))


  def _resume(self):

    if self._mode == 't':

      self.train_ep_rews = self._read_npy("train_ep_rews.npy")
      self.train_ep_lens = self._read_npy("train_ep_lens.npy")

      data = self._read_json("train_stats_summary.json")

      self.train_ep_id  = data["train_episodes"]
      self.train_steps  = data["train_steps"]
      self.env_steps    = data["total_env_steps"]
      self.train_stats["best_mean_rew"] = data["best_mean_rew"]


    if self._mode == 'e':

      self.eval_ep_rews = self._read_npy("eval_ep_rews.npy")
      self.eval_ep_lens = self._read_npy("eval_ep_lens.npy")
      self.eval_ep_inds = self._read_npy("eval_ep_inds.npy")
      self.eval_scores_rews   = self._read_npy("eval_scores_rews.npy")
      self.eval_scores_steps  = self._read_npy("eval_scores_steps.npy")

      data = self._read_json("eval_stats_summary.json")

      self.eval_ep_id   = data["eval_episodes"]
      self.eval_steps   = data["eval_steps"]
      self.env_steps    = data["total_env_steps"]
      self.eval_stats["best_mean_rew"]  = data["best_mean_rew"]
      self.eval_stats["best_score"]     = data["best_score"]


  @property
  def mode(self):
    return self._mode


  @property
  def episode_id(self):
    return self.train_ep_id if self._mode == 't' else self.eval_ep_id


  @property
  def total_steps(self):
    return (self.train_steps if self._mode == 't' else self.eval_steps) + self.ep_steps


  @property
  def mean_ep_rew(self):
    if self._mode == 't':
      return self._stats_mean(self.train_ep_rews)
    else:
      return self._stats_mean(self.eval_ep_rews)


  @property
  def eval_score(self):
    assert self._mode == 'e'
    return self.eval_stats["score_mean"]


  @property
  def episode_rewards(self):
    if self._mode == 't':
      return list(self.train_ep_rews)
    else:
      return list(self.eval_ep_rews)


  @property
  def episode_lens(self):
    if self._mode == 't':
      return list(self.train_ep_lens)
    else:
      return list(self.eval_ep_lens)


  def start_eval_run(self):
    """Signal that eval run begins. Resets the eval temporary data"""
    assert not self._in_eval_run
    self._in_eval_run = True

    self.eval_run_rews = []
    self.eval_run_lens = []

    self.eval_stats["score_mean"]     = -float("inf")
    self.eval_stats["score_std"]      = float("nan")
    self.eval_stats["score_episodes"] = 0


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
      score_mean = -float("inf")
      score_std  = float("nan")

    self.eval_stats["score_mean"]     = score_mean
    self.eval_stats["score_std"]      = score_std
    self.eval_stats["score_episodes"] = episodes

    if score_mean > self.eval_stats["best_score"]:
      self.eval_stats["best_score"] = score_mean
      best_agent = True
    else:
      best_agent = False

    # Append the evaluation score data
    self.eval_scores_rews.append(score_mean)
    self.eval_scores_steps.append(t)

    # Append the episode statistics
    self.eval_ep_rews += self.eval_run_rews
    self.eval_ep_lens += self.eval_run_lens
    self.eval_ep_inds.append(len(self.eval_ep_rews))

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
