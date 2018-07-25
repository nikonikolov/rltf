# RLTF
Reinforcement Learning Library in TensorFlow and OpenAI gym

**Disclaimer: This is work in progress and the library structure is likely to change. Documentation coming soon.**

## About

The goal of this library is to provide implementations of standard RL
algorithms. The framework is designed for research purposes. Some of its
current distinctive features are:
- Unified framework for algorithm implementation and reusable modules
- Easy control over algorithm behavior
- Detailed logs of hyperparameters, training and evaluation scores
- Detailed TensorBoard plots to help debugging. Episode videos recorded with plots of network output
- Saving network weights, train/eval scores or any additional custom metrics
- Restoring the training process from where it stopped, retraining on a new task or fine-tuning

All currently-implemented algorithms achieve competitive performance with results
reported in the original papers. While all implementations are as close as
possible to the ones reported in the original paper, sometimes there might be
very slight differences for the purpose of improving results.


## Installation

### Dependencies
- Python >= 3.5
- Tensorflow >= 1.6.0
- OpenAI gym >= 0.9.6
- opencv-python (either pip package or OpenCV library with python bindings)
- matplotlib (with TkAgg backend)

### Install
```
git clone https://github.com/nikonikolov/rltf.git
```
pip package coming soon


## Implemented Algorithms

| Algorithm                                                 | Model                                          | Agent   |
| ---                                                       | ---                                            | ---     |
| [DQN](https://www.nature.com/articles/nature14236)        | [DQN](rltf/models/dqn.py)                      | [dqn_agent.py](rltf/agents/dqn_agent.py)   |
| [Double DQN](https://arxiv.org/abs/1509.06461)            | [DDQN](rltf/models/ddqn.py)                    | [dqn_agent.py](rltf/agents/dqn_agent.py)   |
| [Dueling DQN](https://arxiv.org/abs/1511.06581)           | next                                           | next                                       |
| [C51](https://arxiv.org/abs/1707.06887)                   | [C51](rltf/models/c51.py)                      | [dqn_agent.py](rltf/agents/dqn_agent.py)   |
| [QR-DQN](https://arxiv.org/abs/1710.10044)                | [QRDQN](rltf/models/qr_dqn.py)                 | [dqn_agent.py](rltf/agents/dqn_agent.py)   |
| [DDPG](https://arxiv.org/abs/1509.02971)                  | [DDPG](rltf/models/ddpg.py)                    | [ddpg_agent.py](rltf/agents/ddpg_agent.py) |
| [NAF](https://arxiv.org/abs/1603.00748)                   | next                                           | next                                       |
| [Bootstrapped DQN](https://arxiv.org/pdf/1602.04621.pdf)  | [BstrapDQN](rltf/models/bstrap_dqn.py)         | [dqn_agent.py](rltf/agents/dqn_agent.py)   |
| [Bootstrapped UCB](https://arxiv.org/pdf/1706.01502.pdf)  | [BstrapDQN_UCB](rltf/models/bstrap_dqn.py)     | [dqn_agent.py](rltf/agents/dqn_agent.py)   |
| [DQN Ensemble](https://arxiv.org/pdf/1706.01502.pdf)      | [BstrapDQN_Ensemble](rltf/models/bstrap_dqn.py)| [dqn_agent.py](rltf/agents/dqn_agent.py)   |
| [PPO](https://arxiv.org/abs/1707.06347)                   | next                                           | next                                       |

Other algorithms which are also coming soon:
- [TRPO](https://arxiv.org/abs/1502.05477)
- [REINFORCE]()
- ...

## Structure

An algorithm is composed of two parts: `Agent` and `Model`

### Agent
- Should inherit from the [`Agent`](rltf/agents/agent.py) class
- Provides communication interface between the [`Model`](rltf/models/model.py) and the environment
- Responsible for
  - Stepping the environment
  - Running a training step
  - Saving data in the replay buffer (if any)

### Model
- Should inherit from the [`Model`](rltf/models/model.py) class
- A passive component which only implements the Tensorflow computation graph for the network
- Implements all network related operations and exposes input and output Tensors
- Controlled by the [`Agent`](rltf/agents/agent.py) during training and evaluation


## Running Examples

After running any of the examples below, the relevant data will be saved in the
directory `trained_models/<model-name>/<env-id>_<date>_<time>`. For example:
`trained_models/dqn/PongNoFrameskip-v4_2018-03-31_19.27.33/`. The directory will
contain:
- `run.log` - Log file with all logs from the training run
- `params.txt` - Log containing the values of all parameter used
- `git.diff` - Diff file with uncommited git changes
- `env_monitor/data/` - data for the training and evaluation statistics. Can be used for plots later
- `env_monitor/*.mp4` - video recordings of episodes, if any were made
- `tf/best_agent/` - the network weights which produced best performance
- `tf/tb/` - TensorBoard files with training and evaluation data
- `tf/` - the latest network weights
- `buffer/` - the latest data of the replay buffer (if saved)


### DQN and derivatives

To see all configurable hyperparameters, run:
```
python3 -m examples.run_dqn_agent --help
```

Note that from the Atari environments only `<AtariEnv>NoFrameskip-v4` and `<AtariEnv>NoFrameskip-v0` are currently supported. To enable other versions of the gym environments, you need to select the correct wrappers from [atari.py](rltf/envs/atari.py).

#### DQN
```
python3 -m examples.run_dqn_agent --model DQN --env-id PongNoFrameskip-v4
```

#### DDQN
```
python3 -m examples.run_dqn_agent --model DDQN --env-id PongNoFrameskip-v4
```

#### Bootstrapped DQN
```
python3 -m examples.run_dqn_agent --model BstrapDQN --n-heads 10 --env-id PongNoFrameskip-v4
```

#### Bootstrapped UCB
```
python3 -m examples.run_dqn_agent --model BstrapDQN_UCB --n-heads 10 --n-stds 0.1 --env-id PongNoFrameskip-v4
```

#### C51
```
python3 -m examples.run_dqn_agent --model C51 --env-id PongNoFrameskip-v4
```

#### QR-DQN
```
python3 -m examples.run_dqn_agent --model QRDQN --env-id PongNoFrameskip-v4
```

### DDPG
To see all configurable hyperparameters, run:
```
python3 -m examples.run_ddpg_agent --help
```
An example configuration is:
```
python3 -m examples.run_ddpg_agent --model DDPG --env-id RoboschoolHopper-v1 --critic-reg 0.0 --sigma 0.5 --max-ep-steps 2500 --noise-decay 500000
```
