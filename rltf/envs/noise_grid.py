import numpy as np
import gym

from gym.utils import seeding

class NoiseGrid(gym.Env):
  """NoiseGrid environment

  n x n grid world. Starting state is a zero state outside the grid.
  From the start, you can choose to enter a specific column and you stay in
  that column for the rest of the episode. Each action after that can take you
  to any row in the grid (but you stay in the same column). The rewards in column
  k are given according to the formula `r_k + N(0,1) * row`. Each observation is
  a 1-hot encoded vector of the columns.

  If you choose action 0 when in a state, you will stay in the same row but get 0 reward
  """

  # TRY:
  #   - Add another entry in the observation which tells you the current row as a simple number
  #     - Remember to change `high` and shape of the Box space
  #   - If you want to allow moving only 1-cell per step, you can simply use the action to indicate
  #     whether you want to move up or down, relative to the current cell

  def __init__(self, n=4, ep_len=None):
    self.n = n            # Grid size
    # self.n = n            # Number of grid columns
    # self.k = k+1          # Number of grid rows + the 0 state
    self.ep_len = n+1 if ep_len is None else ep_len
    self.action_space = gym.spaces.Discrete(self.n+1)
    self.observation_space  = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)
    self.state = 0        # Start at the 0 state
    self.steps = 0        # Start at the 0 state
    self.rewards = np.arange(0, self.n+1, 1) # Array of size n+1
    self.delta = 1
    self.regret = 0
    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    assert self.action_space.contains(action)

    if self.steps == 0:
      self.state = action

    if action == 0:
      reward = 0
      regret = self.rewards[self.n]
      regret += self.delta * self.n if self.steps > 0 else 0
    elif self.steps == 0:
      reward = self.rewards[self.state] + self.np_random.randn()
      regret = self.rewards[self.state] - self.rewards[self.n]
    else:
      reward = self.rewards[self.state] + self.delta * action + self.np_random.randn() * action
      regret = self.rewards[self.n] - self.rewards[self.state] + self.delta * (self.n - action)
    done = True if self.steps >= self.ep_len else False

    self.steps += 1
    self.regret += regret

    return self._encode_state(), reward, done, dict(regret_t=regret, total_regret=self.regret)

  def reset(self):
    self.state = 0
    self.steps = 0
    return self._encode_state()

  def _encode_state(self):
    state = np.zeros(shape=self.n)
    if self.state > 0:
      state[self.state-1] = 1
    return state
