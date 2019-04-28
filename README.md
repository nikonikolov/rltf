# RLTF: Reinforcement Learning in TensorFlow
RLTF is a research framework that provides high-quality implementations of common Reinforcement Learning algorithms. It also allows fast-prototyping and benchmarking of new methods.

**Status**: This work is under active development (breaking changes might occur).

## Implemented Algorithms

| Algorithm                                                 | Model                                           | Agent                                  |
| ---                                                       | ---                                             | ---                                    |
| [DQN](https://www.nature.com/articles/nature14236)        | [DQN](rltf/models/dqn.py)                       | [AgentDQN](rltf/agents/dqn_agent.py)   |
| [Double DQN](https://arxiv.org/abs/1509.06461)            | [DDQN](rltf/models/ddqn.py)                     | [AgentDQN](rltf/agents/dqn_agent.py)   |
| [Dueling DQN](https://arxiv.org/abs/1511.06581)           | next                                            | next                                   |
| [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) | next                                    | next                                   |
| [C51](https://arxiv.org/abs/1707.06887)                   | [C51](rltf/models/c51.py)                       | [AgentDQN](rltf/agents/dqn_agent.py)   |
| [QR-DQN](https://arxiv.org/abs/1710.10044)                | [QRDQN](rltf/models/qr_dqn.py)                  | [AgentDQN](rltf/agents/dqn_agent.py)   |
| [Bootstrapped DQN](https://arxiv.org/pdf/1602.04621.pdf)  | [BstrapDQN](rltf/models/bstrap_dqn.py)          | [AgentDQN](rltf/agents/dqn_agent.py)   |
| [Bootstrapped UCB](https://arxiv.org/pdf/1706.01502.pdf)  | [DQN_UCB](rltf/models/dqn_ucb.py)               | [AgentDQN](rltf/agents/dqn_agent.py)   |
| [DQN Ensemble](https://arxiv.org/pdf/1706.01502.pdf)      | [DQN_Ensemble](rltf/models/dqn_ensemble.py)     | [AgentDQN](rltf/agents/dqn_agent.py)   |
| [BDQN](https://arxiv.org/abs/1802.04412)                  | [BDQN](rltf/models/bdqn.py)                     | [AgentBDQN](rltf/agents/dqn_agent.py)  |
| [DQN-IDS](https://arxiv.org/abs/1812.07544)               | [DQN-IDS](rltf/models/dqn_ids.py)               | [AgentDQN](rltf/agents/dqn_agent.py)   |
| [C51-IDS](https://arxiv.org/abs/1812.07544)               | [C51-IDS](rltf/models/c51_ids.py)               | [AgentDQN](rltf/agents/dqn_agent.py)   |
| [DDPG](https://arxiv.org/abs/1509.02971)                  | [DDPG](rltf/models/ddpg.py)                     | [AgentDDPG](rltf/agents/ddpg_agent.py) |
| [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) | [REINFORCE](rltf/models/reinforce.py)           | [AgentPG](rltf/agents/pg_agent.py)     |
| [PPO](https://arxiv.org/abs/1707.06347)                   | [PPO](rltf/models/ppo.py)                       | [AgentPPO](rltf/agents/ppo_agent.py)   |
| [TRPO](https://arxiv.org/abs/1502.05477)                  | [TRPO](rltf/models/trpo.py)                     | [AgentTRPO](rltf/agents/trpo_agent.py) |


Coming additions:
 - MPI support for policy gradients
 - Dueling DQN
 - Prioritized Experience Replay
 - n-step returns
 - Rainbow


## Reproducibility and Known Issues
Implemented models are able to achieve comparable results to the ones reported
in the corresponding papers. With tiny exceptions, all implementations should be
equivalent to the ones described in the original papers.

Implementations known to misbehave:
- QR-DQN (in progress)


## About

The goal of this framework is to provide stable implementations of standard
RL algorithms and simultaneously enable fast prototyping of new methods.
Some important features include:
- Exact reimplementation and competitive performance of original papers
- Unified and reusable modules
- Clear hierarchical structure and easy code control
- Efficient GPU utilization and fast training
- Detailed logs of hyperparameters, train and eval scores, git diff, TensorBoard visualizations
- Episode video recordings with plots of network outputs
- Compatible with OpenAI gym, MuJoCo, PyBullet and Roboschool
- Restoring the training process from where it stopped, retraining on a new task, fine-tuning


## Installation

### Dependencies
- Python >= 3.5
- Tensorflow >= 1.6.0
- OpenAI gym >= 0.9.6
- opencv-python (either pip package or OpenCV library with python bindings)
- matplotlib (with TkAgg backend)
- pybullet (optional)
- roboschool (optional)

### Install
```
git clone https://github.com/nikonikolov/rltf.git
```
pip package coming soon

## Documentation
For brief documentation see [docs/](docs/).

If you use this repository for you research, please cite:
```
@misc{rltf,
  author = {Nikolay Nikolov},
  title = {RLTF: Reinforcement Learning in TensorFlow},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nikonikolov/rltf}},
}
```
