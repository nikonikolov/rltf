import argparse
import tensorflow as tf

from rltf.agents        import AgentDDPG
from rltf.env_wrappers  import wrap_deepmind_ddpg
from rltf.exploration   import OrnsteinUhlenbeckNoise
from rltf.models        import DDPG
from rltf.optimizers    import OptimizerConf
from rltf.optimizers    import AdamGradClipOptimizer
from rltf.schedules     import ConstSchedule

from rltf import run_utils as rltfru


def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--env-id',       required=True,  type=str,   help='full environment name')
  # parser.add_argument('--model',        required=True,  type=str,   choices=["DDPG", "QRDDPG"])

  parser.add_argument('--actor-lr',     default=1e-4,   type=float, help='actor learn rate',)
  parser.add_argument('--critic-lr',    default=1e-3,   type=float, help='critic learn rate')
  parser.add_argument('--tau',          default=0.001,  type=float, help='target soft update weight')
  parser.add_argument('--sigma',        default=0.2,    type=float, help='ornsein uhlenbeck sigma')
  parser.add_argument('--theta',        default=0.15,   type=float, help='ornsein uhlenbeck theta')
  parser.add_argument('--critic_reg',   default=0.02,   type=float, help='network weight regularization')
  parser.add_argument('--batch-size',   default=None,   type=float, help='batch size')
  parser.add_argument('--adam-epsilon', default=None,   type=float, help='expsilon for Adam optimizer')
  
  parser.add_argument('--train-freq',   default=1,      type=int,   help='learn frequency')
  parser.add_argument('--seed',         default=0,      type=int,   help='seed')
  parser.add_argument('--huber-loss',   default=False,  type=bool,  help='use huber loss')
  parser.add_argument('--grad-clip',    default=None,   type=float, help='value to clip gradinets to')
  parser.add_argument('--extra-info',   default="",     type=str,   help='extra info to log')

  parser.add_argument('--save',         default=False,  type=bool,  help='save model')
  parser.add_argument('--save-video',   default=True,   type=bool,  help='save gym videos')
  parser.add_argument('--video-freq',   default=500,    type=int,
                      help='period in number of episodes at which to record videos')
  
  args = parser.parse_args()
  
  args.model = DDPG

  if args.grad_clip is not None:
    assert args.grad_clip > 0
    assert not args.huber_loss

  return args


def main():

  args = parse_args()

  # Get the model-specific settings
  model_type = args.model

  # Set the model-specific keyword arguments
  model_kwargs = dict(
    critic_reg=args.critic_reg,
    tau=args.tau,
    gamma=0.99,
    huber_loss=args.huber_loss,
  )

  # Get the model directory path
  model_dir = rltfru.make_model_dir(model_type, args.env_id)

  # Create the environment
  env = rltfru.make_env(args.env_id, args.seed, model_dir, args.save_video, args.video_freq)
  env = wrap_deepmind_ddpg(env)

  # Set additional arguments
  if args.batch_size is None:
    args.batch_size = 16 if len(env.observation_space.shape) == 3 else 64
  if args.adam_epsilon is None:
    args.adam_epsilon = 0.01 / float(args.batch_size)

  # Set learning rates and optimizer configuration
  actor_lr  = ConstSchedule(args.actor_lr)
  critic_lr = ConstSchedule(args.critic_lr)

  if args.grad_clip is None:
    actor_opt_conf  = OptimizerConf(tf.train.AdamOptimizer, actor_lr,  epsilon=args.adam_epsilon)
    critic_opt_conf = OptimizerConf(tf.train.AdamOptimizer, critic_lr, epsilon=args.adam_epsilon)
  else:
    opt_args = dict(epsilon=args.adam_epsilon, grad_clip=args.grad_clip)
    actor_opt_conf  = OptimizerConf(AdamGradClipOptimizer, actor_lr,  **opt_args)
    critic_opt_conf = OptimizerConf(AdamGradClipOptimizer, critic_lr, **opt_args)

  # Create the exploration noise
  action_noise = OrnsteinUhlenbeckNoise(mu=0, sigma=args.sigma, theta=args.theta)


  # Set the Agent class keyword arguments
  agent_kwargs = dict(
    env=env,
    train_freq=args.train_freq,
    start_train=1000,
    max_steps=int(2.5e6),
    batch_size=args.batch_size,
    model_dir=model_dir,
    save=args.save,
  )

  ddpg_agent_kwargs = dict(
    model_type=model_type, 
    model_kwargs=model_kwargs,
    actor_opt_conf=actor_opt_conf,
    critic_opt_conf=critic_opt_conf,
    action_noise=action_noise,
    update_target_freq=1,
    memory_size=int(1e6),
    obs_hist_len=1,
  )

  kwargs = {**ddpg_agent_kwargs, **agent_kwargs}

  # Log the parameters for model
  log_info = [("seed", args.seed), ("extra_info", args.extra_info)]
  log_info += kwargs.items()
  rltfru.log_params(model_dir, log_info)

  # Create the agent
  ddpg_agent = AgentDDPG(**kwargs)

  # Build the agent and the TF graph
  ddpg_agent.build()

  # Train the agent
  ddpg_agent.train()

  # Close on exit
  ddpg_agent.close()
  env.close()

if __name__ == "__main__":
  main()
