## Structure Overview

All algorithms are composed of two parts: `Agent` and `Model`

### Agent
- Should inherit from the [`Agent`](rltf/agents/agent.py) class
- Provides communication interface between the [`Model`](rltf/models/model.py) and the environment
- Executes the exact training procedure
- Responsible for
  - Stepping the environment
  - Running a training step
  - Storing experience for training

### Model
- Should inherit from the [`Model`](rltf/models/model.py) class
- A passive component which only implements the Tensorflow computation graph for the network
- Implements forward and backward network pass, exposes useful input and output Tensors and Operations
- Controlled by the [`Agent`](rltf/agents/agent.py) during training and evaluation

-------------------------------------------------------------------------------

## Execution

The data for separate runs is stored on disk under the template directory path
`trained_models/<model-name>/<env-id>_<date>_<time>`. For example:
`trained_models/dqn/PongNoFrameskip-v4_2018-03-31_19.27.33/`.

Each run directory contains"
- `run.log` - Log file, including git branch and hash, all hyperparameters, train and eval metrics
- `git.diff` - Diff file with uncommited git changes at the time of launch
- `monitor/data/` - `numpy` data with train and eval statistics. Can be used for custom plots
- `monitor/videos/` - video recordings of episodes, if any were made
- `monitor/videos/` - TensorBoard files
- `snapshots/latest/` - latest training checkpoint
- `snapshots/best/` - checkpoint for which produced the best eval score
- `buffer/` - latest state and data of the replay buffer (if saved)

Every saved model can be restored and training continued as if it never stopped
(the only difference is state of the random number generators). Additionally, model
variables can be used for initializing or fune-tuning a new model on a new task.

-------------------------------------------------------------------------------

## Usage

### DQN Family

Usage:
```bash
python3 -m examples.run_dqn_agent --model=<model> --env_id=<env>
```
Allowed models include: `DQN, DDQN, C51, QRDQN, BstrapDQN, DQN_UCB,
DQN_Ensemble, DQN_IDS, C51_IDS, QRDQN_IDS, BDQN, BDQN_TS, BDQN_UCB, BDQN_IDS`

From the Atari family only `<AtariEnv>NoFrameskip-v4` and `<AtariEnv>NoFrameskip-v0`
are currently supported. To enable other versions of the gym environments, you need
to select the correct wrappers from [atari.py](rltf/envs/atari.py).

### Policy Gradients Family

Usage:
```bash
python3 -m examples.run_pg_agent --model=<model> --env_id=<env>
```
Allowed models include: `DDPG, REINFORCE, TRPO, PPO`

### Command Line Arguments
All default hyperparameters are located in [`rltf/cmdutils/defaults.py`](rltf/cmdutils/defaults.py).
All of these can be directly overriden from the command line, for example:

```bash
python3 -m examples.run_dqn_agent --model=DQN --env_id=PongNoFrameskip-v4 --log_period=10000
```

One can also directly override the arugments for custom python objects.
For example, override the learning rate of `rltf.optimizers.OptimizerConf`:

```bash
python3 -m examples.run_dqn_agent --model=DQN --env_id=PongNoFrameskip-v4 --opt_conf.learn_rate=5e-5
```

Additionally entire custom objects can be provided on the command line,
e.g. using `AdamOptimizer` instead of `RMSPropOptimizer`:

```bash
python3 -m examples.run_dqn_agent --model=DQN --env_id=PongNoFrameskip-v4 --opt_conf='OptimizerConf(opt_type=tf.train.AdamOptimizer, learn_rate=5e-5, epsilon=.01/32)'
```

### Restoring a Model

```bash
python3 -m examples.run_dqn_agent --model=DQN --env_id=PongNoFrameskip-v4 --restore=trained_models/dqn/PongNoFrameskip-v4_2018-12-28_12.44.17
```
This will restore the latest checkpoint in `--restore` and
continue training from where it left off.

**NOTE**: You still need to make sure to provide the same `model`, `env_id`
and hyperparameters that were used to launch the original model. Restoring
the model won't create a new directory, but use the one provided in `--restore`.


### Evaluating a Model

One can use the same scripts to evaluate a model, but needs to provide additional arguments, e.g.
```bash
python3 -m examples.run_dqn_agent --model=DQN --env_id=PongNoFrameskip-v4 --load_model=trained_models/dqn/PongNoFrameskip-v4_2018-12-28_12.44.17 --n_plays=10 --eval_len=100000
```
The above will load the best checkpoint from in `--load_model` and
perform `--n_plays` evaluation runs, each of length `--eval_len`. The data
will be saved in the a sub-directory called `play` inside the `--load_model` directory.


### Using Model variables

To use the variables of a trained model for initializing a new model, use:
```bash
python3 -m examples.run_dqn_agent --model=DQN --env_id=PongNoFrameskip-v4 --load_model=trained_models/dqn/PongNoFrameskip-v4_2018-12-28_12.44.17
```

This will fetch the best checkpoint in `--load_model` and use the checkpoint
values for initialization. Note that variable names of the loaded model must match
variable names of the built computational graph. Additionally, one can filter which
variables are restored using a regex and the `--load_regex` argument. All variable
names which are matched by the regex will be loaded; the rest will be randomly initialized.
