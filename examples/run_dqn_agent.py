import tensorflow as tf

from rltf.agents        import AgentDQN
from rltf.envs          import wrap_dqn
from rltf.models        import BstrapDQN
from rltf.models        import DDQN
from rltf.models        import DQN
from rltf.models        import C51
from rltf.models        import QRDQN
from rltf.optimizers    import OptimizerConf
from rltf.schedules     import ConstSchedule
from rltf.schedules     import PiecewiseSchedule
from rltf.utils         import rltf_log
from rltf.utils         import maker
from rltf.utils         import cmdargs


def parse_args():

  model_types = ["DQN", "DDQN", "C51", "QRDQN", "BstrapDQN"]
  s2b         = cmdargs.str2bool

  args = [
    ('--env-id',        dict(required=True,  type=str,   help='full environment name')),
    ('--model',         dict(required=True,  type=str,   choices=model_types)),

    ('--learn-rate',    dict(default=5e-5,   type=float, help='learn rate',)),
    ('--batch-size',    dict(default=32,     type=int,   help='batch size for training the net',)),
    ('--memory-size',   dict(default=10**6,  type=int,   help='size of the replay buffer',)),
    ('--adam-epsilon',  dict(default=.01/32, type=float, help='epsilon for Adam optimizer')),
    ('--n-heads',       dict(default=10,     type=int,   help='number of heads for BstrapDQN')),
    ('--explore-decay', dict(default=10**6,  type=int,   help='# steps to decay e-greedy; if <=0, epsilon=0')),
    ('--epsilon-eval',  dict(default=0.001,  type=float, help='epsilon value during evaluation')),

    ('--warm-up',       dict(default=50000,  type=int,   help='# steps before training starts')),
    ('--train-freq',    dict(default=4,      type=int,   help='learn frequency')),
    ('--update-freq',   dict(default=10000,  type=int,   help='how often to update target')),
    ('--stop-step',     dict(default=5*10**7,type=int,   help='steps to run the agent for')),
    ('--huber-loss',    dict(default=True,   type=s2b,   help='use huber loss')),
    # ('--grad-clip',     dict(default=None,   type=float, help='value to clip gradient norms to')),

    ('--eval-freq',     dict(default=10**6,  type=int,   help='how often to evaluate model')),
    ('--eval-len',      dict(default=500000, type=int,   help='for how many steps to eval each time')),
  ]

  return cmdargs.parse_args(args)


def make_agent():

  args = parse_args()

  # Get the model directory path
  if args.restore_model is None:
    model_dir   = maker.make_model_dir(args.model, args.env_id)
    restore_dir = args.reuse_model
  else:
    model_dir   = args.restore_model
    restore_dir = args.restore_model

  # Configure loggers
  rltf_log.conf_logs(model_dir)

  # Get the model-specific settings
  model = eval(args.model)

  if   args.model in ["DQN", "DDQN"]:
    model_kwargs  = dict(huber_loss=args.huber_loss)
  elif args.model == "BstrapDQN":
    model_kwargs  = dict(huber_loss=args.huber_loss, n_heads=args.n_heads)
  elif args.model == "C51":
    model_kwargs  = dict(V_min=-10, V_max=10, N=51)
  elif args.model == "QRDQN":
    model_kwargs  = dict(N=200, k=int(args.huber_loss))

  model_kwargs["gamma"] = args.gamma


  # Create the environment
  env = maker.make_env(args.env_id, args.seed, model_dir, args.video_freq)
  env = wrap_dqn(env)

  # Set the learning rate schedule
  learn_rate = ConstSchedule(args.learn_rate)

  # Cteate the optimizer configs
  opt_conf = OptimizerConf(tf.train.AdamOptimizer, learn_rate, epsilon=args.adam_epsilon)

  # Create the exploration schedule
  if args.explore_decay > 0:
    exploration = PiecewiseSchedule([(0, 1.0), (args.explore_decay, 0.01)], outside_value=0.01)
  else:
    exploration = ConstSchedule(0.0)


  # Set the Agent class keyword arguments
  agent_kwargs = dict(
    env=env,
    train_freq=args.train_freq,
    warm_up=args.warm_up,
    stop_step=args.stop_step,
    eval_freq=args.eval_freq,
    eval_len=args.eval_len,
    batch_size=args.batch_size,
    model_dir=model_dir,
    log_freq=args.log_freq,
    save_freq=args.save_freq,
    restore_dir=restore_dir,
    confirm_kill=args.confirm_kill,
    reuse_regex=args.reuse_regex,
  )

  dqn_agent_kwargs = dict(
    model=model,
    model_kwargs=model_kwargs,
    opt_conf=opt_conf,
    exploration=exploration,
    update_target_freq=args.update_freq,
    memory_size=args.memory_size,
    obs_len=4,
    epsilon_eval=args.epsilon_eval,
  )

  kwargs = {**dqn_agent_kwargs, **agent_kwargs}

  # Log the parameters for model
  log_info = [("seed", args.seed), ("extra_info", args.extra_info)]
  log_info += kwargs.items()
  rltf_log.log_params(log_info, args)

  # Create the agent
  dqn_agent = AgentDQN(**kwargs)

  return dqn_agent, args


def main():
  # Create the agent
  dqn_agent, args = make_agent()

  # Build the agent and the TF graph
  dqn_agent.build()

  # Train or eval the agent
  if args.mode == 'train':
    dqn_agent.train()
  else:
    dqn_agent.eval()

  # Close on exit
  dqn_agent.close()


if __name__ == "__main__":
  main()
