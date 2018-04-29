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
- TensorBoard plots. Configurable matplotlib plots coming soon
- Saving network weights, train/eval scores or any additional custom metrics
- Restoring the training process from where it stopped or retraining on a new task

All currently-implemented algorithms achieve competitive performance with results
reported in the original papers. While all implementations are as close as
possible to the ones reported in the original paper, sometimes there might be
very slight differences for the purpose of improving results.


## Installation

### Dependencies
- Python >= 3.5
- Tensorflow >= 1.4.0
- OpenAI gym >= 0.9.6
- opencv-python (pip package or OpenCV library)
- matplotlib (with TkAgg backend)

### Install
```
git clone https://github.com/nikonikolov/rltf.git
```
pip package coming soon


## Implemented Algorithms

| Algorithm     | Model                             | Agent                                       | Orignal Paper |
| ---           | ---                               | ---                                         | --- |
| DQN           | [dqn.py](rltf/models/dqn.py)      | [dqn_agent.py](rltf/agents/dqn_agent.py)    | [DQN](https://www.nature.com/articles/nature14236) |
| Double DQN    | [ddqn.py](rltf/models/ddqn.py)    | [dqn_agent.py](rltf/agents/dqn_agent.py)    | [Double DQN](https://arxiv.org/abs/1509.06461) |
| Dueling DQN   | next                              | next                                        | [Dueling DQN](https://arxiv.org/abs/1511.06581) |
| C51           | [c51.py](rltf/models/c51.py)      | [dqn_agent.py](rltf/agents/dqn_agent.py)    | [C51](https://arxiv.org/abs/1707.06887) |
| QR-DQN        | [qrdqn.py](rltf/models/qrdqn.py)  | [dqn_agent.py](rltf/agents/dqn_agent.py)    | [QR-DQN](https://arxiv.org/abs/1710.10044) |
| DDPG          | [ddpg.py](rltf/models/ddpg.py)    | [ddpg_agent.py](rltf/agents/ddpg_agent.py)  | [DDPG](https://arxiv.org/abs/1509.02971) |
| NAF           | next                              | next                                        | [NAF](https://arxiv.org/abs/1603.00748) |
| Bootstrapped DQN | [bstrap_dqn.py](rltf/models/bstrap_dqn.py) | [dqn_agent.py](rltf/agents/dqn_agent.py) | [Bootstrapped DQN](https://arxiv.org/pdf/1602.04621.pdf) |
| PPO           | next                              | next                                        | [PPO](https://arxiv.org/abs/1707.06347) |

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
- `env_monitor/data/` - data for the training and evaluation statistics. Can be used for plots later
- `env_monitor/*.mp4` - video recordings of episodes, if any were made
- `tf/tb_train/` - TensorBoard file with training data
- `tf/tb_eval/` - TensorBoard file with evaluation data
- `tf/` - will also contain the saved graph, which can be restored later


### DDPG
To see configurable parameters, run:
```
python3 -m examples.run_dqn_agent --help
```
An example configuration is:
```
python3 -m examples.run_ddpg_agent --model DDPG --env-id RoboschoolHopper-v1 --critic-reg 0.0 --sigma 0.5 --max-ep-steps 2500 --noise-decay 500000
```

### DQN, C51, QR-DQN, Double DQN
To see configurable parameters, run:
```
python3 -m examples.run_dqn_agent --help
```

At the moment `run_dqn_agent.py` enforces only Atari environments and image
observations. It also requires that the environment is of the `NoFrameskip-v4`
OpenAI gym family. The `NoFrameskip-v4` environments (together with some
additional wrappers) replicate the training process desribed in the orginal DQN
Nature paper. To make modifications, you need to add/remove some wrappers (see
[atari.py](rltf/envs/atari.py)).
