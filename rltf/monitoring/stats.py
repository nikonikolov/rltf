import json
import logging
import os
import time
import numpy as np

from gym        import error
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

    # Training statistics
    self.train_ep_rews = []
    self.train_ep_lens = []
    self.train_steps   = 0      # Total number of environment steps in train mode
    self.train_stats   = None   # A dictionary with runtime statistics about training
    self.train_ep_id   = 0

    # Evaluation statistics
    self.eval_ep_rews = []
    self.eval_ep_lens = []
    self.eval_steps   = 0       # Total number of environment steps in eval mode
    self.eval_stats   = None    # A dictionary with runtime statistics about evaluation
    self.eval_ep_id   = 0

    # NOTE: self.train_ep_id and self.eval_ep_id are the episode IDs as seen by the agent.
    # The length of  train_ep_lens and eval_ep_lens is not guaranteed to be the same

    # Runtime variables
    self.ep_reward  = None
    self.ep_steps   = None
    self.env_steps  = 0         # Total number of environment steps in any mode
    self.disabled   = False
    self._mode      = mode      # Running mode: either 't' (train) or 'e' (eval)
    self._new_mode  = mode      # Used to synchronize changes when mode is changed mid-episode

    if not os.path.exists(self.log_dir):
      logger.info('Creating stats directory %s', self.log_dir)
      os.makedirs(self.log_dir)
    else:
      logger.info('Resuming with stats data from %s', self.log_dir)
      self._resume()


  @property
  def mode(self):
    return self._mode


  @mode.setter
  def mode(self, mode):
    if mode not in ['t', 'e']:
      raise error.Error('Invalid mode {}: must be t for training or e for evaluation', mode)
    self._new_mode = mode


  def before_step(self, action):
    assert not self.disabled


  def after_step(self, obs, reward, done, info):
    self.ep_steps   += 1
    self.env_steps  += 1
    self.ep_reward  += reward

    if done:
      self._finish_episode()


  def _finish_episode(self):
    # The mode at the end of the episode is used to determine how to record the statistics
    if self._mode == 't':
      self.train_steps += self.ep_steps
      self.train_ep_lens.append(np.int32(self.ep_steps))
      self.train_ep_rews.append(np.float32(self.ep_reward))
    else:
      self.eval_steps += self.ep_steps
      self.eval_ep_lens.append(np.int32(self.ep_steps))
      self.eval_ep_rews.append(np.float32(self.ep_reward))


  def before_reset(self):
    assert not self.disabled


  def after_reset(self, obs):
    self.ep_steps   = 0
    self.ep_reward  = 0

    # Mode should be changed before episode_id is incremented!
    if self._mode != self._new_mode:
      self._mode = self._new_mode
      logger.info("Monitor mode set to %s", "TRAIN" if self._mode == 't' else "EVAL")

    if self._mode == 't':
      self.train_ep_id += 1
    else:
      self.eval_ep_id  += 1


  def close(self):
    self.disabled = True


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

    self.train_stats = {
      "mean_ep_rew":    float("nan"),
      "std_ep_rew":     float("nan"),
      "ep_len_mean":    float("nan"),
      "ep_len_std":     float("nan"),
      "best_mean_rew":  -float("inf"),
      "best_ep_rew":    -float("inf"),
      "ep_last_stats":  0,
    }

    self.eval_stats = {
      "mean_ep_rew":    float("nan"),
      "std_ep_rew":     float("nan"),
      "ep_len_mean":    float("nan"),
      "ep_len_std":     float("nan"),
      "best_mean_rew":  -float("inf"),
      "best_ep_rew":    -float("inf"),
      "ep_last_stats":  0,
    }

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

    self.train_stats["mean_ep_rew"] = self._stats_mean(self.train_ep_rews)
    self.train_stats["std_ep_rew"]  = self._stats_std(self.train_ep_rews)
    self.train_stats["ep_len_mean"] = self._stats_mean(self.train_ep_lens)
    self.train_stats["ep_len_std"]  = self._stats_std(self.train_ep_lens)
    self.train_stats["best_mean_rew"] = max(self.train_stats["best_mean_rew"],
                                            self.train_stats["mean_ep_rew"])
    best_ep_rew = _stats_max(self.train_ep_rews, self.train_stats["ep_last_stats"])
    self.train_stats["best_ep_rew"] = max(self.train_stats["best_ep_rew"], best_ep_rew)
    self.train_stats["ep_last_stats"] = len(self.train_ep_rews)


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
    stats_logger.info(self.log_info[0][0].format(self.log_info[0][1](t)))
    for s, lambda_v in self.log_info[1:-1]:
      if self._mode == 't' and not s.startswith("| eval/"):
        stats_logger.info(s.format(lambda_v(t)))
      elif self._mode == 'e' and s.startswith("| eval/"):
        stats_logger.info(s.format(lambda_v(t)))
    stats_logger.info(self.log_info[-1][0].format(self.log_info[-1][1](t)))
    stats_logger.info("")


  def save(self):
    """Save the statistics data to disk. Must be manually called"""
    if self.disabled:
      return

    summary_file = os.path.join(self.log_dir, "stats_summary.json")
    data = {
      "total_env_steps":  self.env_steps,
      "train_steps":      self.train_steps,
      "train_episodes":   self.train_ep_id,
      "eval_steps":       self.eval_steps,
      "eval_episodes":    self.eval_ep_id,
    }

    with atomic_write.atomic_write(summary_file) as f:
      json.dump(data, f, indent=4, sort_keys=True)

    if self.train_ep_rews:
      train_rew_file = os.path.join(self.log_dir, "train_ep_rews.npy")
      with atomic_write.atomic_write(train_rew_file, True) as f:
        np.save(f, np.asarray(self.train_ep_rews, dtype=np.float32))

      train_ep_len_file = os.path.join(self.log_dir, "train_ep_lens.npy")
      with atomic_write.atomic_write(train_ep_len_file, True) as f:
        np.save(f, np.asarray(self.train_ep_lens, dtype=np.int32))

    if self.eval_ep_rews:
      eval_rew_file = os.path.join(self.log_dir, "eval_ep_rews.npy")
      with atomic_write.atomic_write(eval_rew_file, True) as f:
        np.save(f, np.asarray(self.eval_ep_rews, dtype=np.float32))

      eval_ep_len_file = os.path.join(self.log_dir, "eval_ep_lens.npy")
      with atomic_write.atomic_write(eval_ep_len_file, True) as f:
        np.save(f, np.asarray(self.eval_ep_lens, dtype=np.int32))


  def _resume(self):
    train_rew_file = os.path.join(self.log_dir, "train_ep_rews.npy")
    if os.path.exists(train_rew_file):
      self.train_ep_rews = list(np.load(train_rew_file))

    eval_rew_file = os.path.join(self.log_dir, "eval_ep_rews.npy")
    if os.path.exists(eval_rew_file):
      self.eval_ep_rews = list(np.load(eval_rew_file))

    train_ep_len_file = os.path.join(self.log_dir, "train_ep_lens.npy")
    if os.path.exists(train_ep_len_file):
      self.train_ep_lens = list(np.load(train_ep_len_file))

    eval_ep_len_file = os.path.join(self.log_dir, "eval_ep_lens.npy")
    if os.path.exists(eval_ep_len_file):
      self.eval_ep_lens = list(np.load(eval_ep_len_file))

    with open(os.path.join(self.log_dir, "stats_summary.json"), 'r') as f:
      data = json.load(f)

    self.train_ep_id  = data["train_episodes"]
    self.eval_ep_id   = data["eval_episodes"]
    self.train_steps  = data["train_steps"]
    self.eval_steps   = data["eval_steps"]
    self.env_steps    = data["total_env_steps"]


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
