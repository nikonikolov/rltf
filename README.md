# rltf
Reinforcement Learning Library in TensorFlow and OpenAI gym

**NOTE: This is work in progress and the core structure of some parts of the library is likely to change**

## About

The goal of this library is to provide standard implementations of core
Reinforcement Learning algorithms. The library is specifically targetted at
research applications and aims to provide reusable constructs for easy
implementations of new algorithms. Furthermore, it includes detailed
hyperparameter and training logs, real-time training metrics plots (currently
in TensorBoard, configurable matplotlib plots coming soon). The code is written
in TensorFlow and supports gym-compatible envornments.

All currently-implemented algorithms achieve competitive performance with the
results reported in the original papers (the default hyperparameters are not
optimal for all environments).


## Installation
```
git clone https://github.com/nikonikolov/rltf.git
```

### Dependencies
- Python >= 3.5
- Tensorflow >= 1.4.0
- OpenAI gym


## Implemented Algorithms

| Algorithm     | Model                     | Agent                           | Orignal Paper |
| ---           | ---                       | ---                             | ---           |  
| DQN           | [](rltf/models/dqn.py)    | [](rltf/agents/dqn_agent.py)    | [DQN](https://www.nature.com/articles/nature14236) |
| Double DQN    | next                      | next                            | [Double DQN](https://arxiv.org/abs/1509.06461) |
| Dueling DQN   | next                      | next                            | [Dueling DQN](https://arxiv.org/abs/1511.06581) |
| C51           | [](rltf/models/c51.py)    | [](rltf/agents/dqn_agent.py)    | [C51](https://arxiv.org/abs/1707.06887) |
| QR-DQN        | [](rltf/models/qrdqn.py)  | [](rltf/agents/dqn_agent.py)    | [QR-DQN](https://arxiv.org/abs/1710.10044) |
| DDPG          | [](rltf/models/ddpg.py)   | [](rltf/agents/ddpg_agent.py)   | [DDPG](https://arxiv.org/abs/1509.02971) |
| NAF           | next                      | next                            | [NAF](https://arxiv.org/abs/1603.00748) |

Other algorithms are also coming in the near future:
- [Soft Q-learning](https://arxiv.org/abs/1702.08165)
- [A3C](https://arxiv.org/pdf/1602.01783.pdf)
- [TRPO](https://arxiv.org/abs/1502.05477)
- [PPO](https://arxiv.org/abs/1707.06347)
- [REINFORCE]()

## Structure

An implemntation of an algorithm is composed of two parts: agent and model

### Agent
- Should inherit from the [Agent](rltf/agents/agent.py) class
- Provides communication between the [Model](rltf/models/model.py) and the environment
- Responsible for stepping the environment and running the train procedure
- Manages the replay buffer (if any)

### Model
- Should inherit from the [Model](rltf/models/model.py) class
- A passive component which only implements the Tensorflow computation graph for the algorithm
- Implements the graph training procedure
- Exposes the graph input and output Tensors so they can be run by the [Agent](rltf/agents/agent.py)


## Running Examples

After running any of the examples below, your logs will be saved in 
`trained_models/<model>/<env-id>_<run-number>`. If you enabled model saving,
the NN and its weights will be saved in the same folder. Furthermore, the
folder will contain:
- `params.txt` - file containing the values of the hyperparameters used
- `run.log` - runtime log of the program
- `tb/` - folder containing the TensorBoard plots of the training process

To see the TensorBoard plots, run:
```
tensorboard --logdir="<path/to/tb/dir"
```
and then go to http://localhost:6006 in your browser

### DDPG
```
python3 -m examples.run_ddpg_agent --model <model-name> --env-id <env-id>
```
For more details run:
```
python3 -m examples.run_dqn_agent --help
```

### DQN, C51, QR-DQN

```
python3 -m examples.run_dqn_agent --model <model-name> --env-id <env-id>
```
For more details run:
```
python3 -m examples.run_dqn_agent --help
```


Note that `run_dqn_agent` enforces only Atari environments. Moreover, it
requires that the environment used is of type `<env-name>NoFrameskip-v4`
(e.g. `PongNoFrameskip-v4`). The `NoFrameskip-v4` gym environments (combined 
with some additional wrappers) are the ones corresponding to the training
process described in the orginal DQN Nature paper. If you want to use other
environment versions, you will need to add or remove some env wrappers 
(see [](rltf/env_wrap/atari.py))
