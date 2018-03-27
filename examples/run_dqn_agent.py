import argparse
import tensorflow as tf

from rltf.agents        import AgentDQN
from rltf.env_wrap      import wrap_deepmind_atari
# from rltf.exploration   import EGreedy
from rltf.models        import BstrapDQN
from rltf.models        import DDQN
from rltf.models        import DQN
from rltf.models        import DQN_IDS_BLR
from rltf.models        import C51
from rltf.models        import QRDQN
from rltf.optimizers    import OptimizerConf
from rltf.optimizers    import AdamGradClipOptimizer
from rltf.schedules     import ConstSchedule
from rltf.schedules     import PiecewiseSchedule
from rltf.utils         import rltf_log
from rltf.utils         import maker
from rltf.utils.cmdargs import str2bool


def parse_args():
  model_choices = ["DQN", "DDQN", "C51", "QRDQN", "BstrapDQN", "DQN_IDS_BLR"]

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--env-id',       required=True,  type=str,       help='full environment name')
  parser.add_argument('--model',        required=True,  type=str,       choices=model_choices)

  parser.add_argument('--learn-rate',   default=None,   type=float,     help='learn rate',)
  parser.add_argument('--adam-epsilon', default=.01/32, type=float,     help='epsilon for Adam optimizer')

  parser.add_argument('--train-freq',   default=4,      type=int,       help='learn frequency')
  parser.add_argument('--start-train',  default=50000,  type=int,       help='step to start training')
  parser.add_argument('--max-steps',    default=10**8,  type=int,       help='steps to run the agent for')
  parser.add_argument('--n-heads',      default=10,     type=int,       help='number of BstrapDQN heads')
  parser.add_argument('--seed',         default=0,      type=int,       help='seed')
  parser.add_argument('--huber-loss',   default=True,   type=str2bool,  help='use huber loss')
  parser.add_argument('--grad-clip',    default=None,   type=float,     help='value to clip gradinets to')
  parser.add_argument('--extra-info',   default="",     type=str,       help='extra info to log')

  parser.add_argument('--save',         default=False,  type=str2bool,  help='save model')
  parser.add_argument('--save-video',   default=True,   type=str2bool,  help='save gym videos')
  parser.add_argument('--video-freq',   default=1000,   type=int,
                      help='period in number of episodes at which to record videos')

  args = parser.parse_args()

  if args.grad_clip is not None:
    assert args.grad_clip > 0
    assert not args.huber_loss

  return args


def main():

  args = parse_args()

  # Get the model directory path
  model_dir = maker.make_model_dir(args.model, args.env_id)

  # Configure loggers
  rltf_log.conf_logs(model_dir)

  # Get the model-specific settings
  if   args.model == "DQN":
    model_type    = DQN
    model_kwargs  = dict(huber_loss=args.huber_loss)
  elif args.model == "DDQN":
    model_type    = DDQN
    model_kwargs  = dict(huber_loss=args.huber_loss)
  elif args.model == "BstrapDQN":
    model_type    = BstrapDQN
    model_kwargs  = dict(huber_loss=args.huber_loss, n_heads=args.n_heads)
  elif args.model == "DQN_IDS_BLR":
    model_type    = DQN_IDS_BLR
    model_kwargs  = dict(huber_loss=args.huber_loss, sigma=1.0, tau=25.0, rho=0.5)
  elif args.model == "C51":
    model_type    = C51
    model_kwargs  = dict(V_min=-10, V_max=10, N=50)
  elif args.model == "QRDQN":
    model_type    = QRDQN
    model_kwargs  = dict(N=200, k=int(args.huber_loss))

  model_kwargs["gamma"] = 0.99


  # Create the environment
  env = maker.make_env(args.env_id, args.seed, model_dir, args.save_video, args.video_freq)
  env = wrap_deepmind_atari(env)

  # Set the learning rate schedule
  if args.learn_rate is None:
    learn_rate = PiecewiseSchedule([(.0, 1e-4), (1e6, 1e-4), (5e6, 5e-5)], outside_value=5e-5)
  else:
    learn_rate = ConstSchedule(args.learn_rate)

  # Cteate the optimizer configs
  if args.grad_clip is None:
    opt_conf = OptimizerConf(tf.train.AdamOptimizer, learn_rate, epsilon=args.adam_epsilon)
  else:
    opt_args = dict(epsilon=args.adam_epsilon, grad_clip=args.grad_clip)
    opt_conf = OptimizerConf(AdamGradClipOptimizer, learn_rate, **opt_args)

  # Create the exploration schedule
  # exploration = PiecewiseSchedule([(0, 1.0), (1e7, 0.01)], outside_value=0.01)
  exploration = PiecewiseSchedule([(0, 1.0), (1e6, 0.1)], outside_value=0.01)

  # Set the Agent class keyword arguments
  agent_kwargs = dict(
    env=env,
    train_freq=args.train_freq,
    start_train=args.start_train,
    max_steps=args.max_steps,
    batch_size=32,
    model_dir=model_dir,
    save=args.save,
  )

  dqn_agent_kwargs = dict(
    model_type=model_type,
    model_kwargs=model_kwargs,
    opt_conf=opt_conf,
    exploration=exploration,
    update_target_freq=10000,
    memory_size=int(1e6),
    obs_hist_len=4,
  )

  kwargs = {**dqn_agent_kwargs, **agent_kwargs}

  # Log the parameters for model
  log_info = [("seed", args.seed), ("extra_info", args.extra_info)]
  log_info += kwargs.items()
  rltf_log.log_params(log_info, args)

  # Create the agent
  dqn_agent = AgentDQN(**kwargs)

  # Build the agent and the TF graph
  dqn_agent.build()

  # Train the agent
  dqn_agent.train()

  # Close on exit
  dqn_agent.close()
  env.close()

if __name__ == "__main__":
  main()
