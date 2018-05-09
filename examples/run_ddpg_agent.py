import numpy      as np
import tensorflow as tf

from rltf.agents        import AgentDDPG
from rltf.envs          import wrap_deepmind_ddpg
from rltf.exploration   import DecayedExplorationNoise
from rltf.exploration   import GaussianNoise
from rltf.exploration   import OrnsteinUhlenbeckNoise
from rltf.models        import DDPG
from rltf.models        import QRDDPG
from rltf.optimizers    import OptimizerConf
from rltf.schedules     import PiecewiseSchedule
from rltf.schedules     import ConstSchedule
from rltf.utils         import rltf_log
from rltf.utils         import maker
from rltf.utils         import cmdargs


def parse_args():

  model_types = ["DDPG", "QRDDPG"]
  noise_types = ["OU", "Gaussian"]
  s2b         = cmdargs.str2bool

  args = [
    ('--env-id',       dict(required=True,  type=str,   help='full environment name')),
    ('--model',        dict(required=True,  type=str,   choices=model_types)),

    ('--N',            dict(default=100,    type=int,   help='number of quantiles for QRDDPG')),
    ('--actor-lr',     dict(default=1e-4,   type=float, help='actor learn rate')),
    ('--critic-lr',    dict(default=1e-3,   type=float, help='critic learn rate')),
    ('--tau',          dict(default=0.001,  type=float, help='weight for soft target update')),
    ('--critic-reg',   dict(default=0.02,   type=float, help='network weight regularization param')),
    ('--batch-size',   dict(default=None,   type=int,   help='batch size')),
    ('--memory-size',  dict(default=10**6,  type=int,   help='size of the replay buffer',)),

    ('--sigma',        dict(default=0.5,    type=float, help='action noise sigma')),
    ('--theta',        dict(default=0.15,   type=float, help='action noise theta (for OU noise)')),
    ('--dt',           dict(default=1e-2,   type=float, help='action noise dt (for OU noise)')),
    ('--noise-type',   dict(default="OU",   type=str,   help='action noise type', choices=noise_types)),
    ('--noise-decay',  dict(default=500000, type=int,   help='action noise decay; \
      # steps to decay noise weight from 1 to 0; if <=0, no decay')),

    ('--warm-up',      dict(default=10000,  type=int,   help='# steps before training starts')),
    ('--update-freq',  dict(default=1,      type=int,   help='how often to update target')),
    ('--train-freq',   dict(default=1,      type=int,   help='training frequency in # steps')),
    ('--stop-step',    dict(default=2500000,type=int,   help='steps to run the agent for')),
    ('--batch-norm',   dict(default=False,  type=s2b,   help='apply batch normalization')),
    ('--obs-norm',     dict(default=False,  type=s2b,   help='normalize observations')),
    ('--huber-loss',   dict(default=False,  type=s2b,   help='use huber loss for critic')),
    ('--reward-scale', dict(default=1.0,    type=float, help='scale env reward')),
    ('--max-ep-steps', dict(default=None,   type=int,   help='max # steps for an episode')),

    ('--eval-freq',    dict(default=500000, type=int,   help='how often to evaluate model')),
    ('--eval-len',     dict(default=50000,  type=int,   help='for how many steps to eval each time')),
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
  rltf_log.conf_logs(model_dir, args.log_lvl)

  # Set the model-specific keyword arguments
  model_kwargs = dict(
    critic_reg=args.critic_reg,
    tau=args.tau,
    gamma=args.gamma,
    huber_loss=args.huber_loss,
    batch_norm=args.batch_norm,
    obs_norm=args.obs_norm,
  )

  # Get the model-specific settings
  if    args.model == "DDPG":
    model = DDPG
  elif  args.model == "QRDDPG":
    model = QRDDPG
    model_kwargs["N"] = args.N


  # Create the environment
  env = maker.make_env(args.env_id, args.seed, model_dir, args.video_freq)
  env = wrap_deepmind_ddpg(env, rew_scale=args.reward_scale, max_ep_len=args.max_ep_steps)

  # Set additional arguments
  if args.batch_size is None:
    args.batch_size = 16 if len(env.observation_space.shape) == 3 else 64

  # Set learning rates and optimizer
  actor_opt_conf  = OptimizerConf(tf.train.AdamOptimizer, ConstSchedule(args.actor_lr))
  critic_opt_conf = OptimizerConf(tf.train.AdamOptimizer, ConstSchedule(args.critic_lr))

  # Create the exploration noise
  mu            = np.zeros(env.action_space.shape, dtype=np.float32)
  sigma         = np.ones(env.action_space.shape,  dtype=np.float32) * args.sigma
  if args.noise_type == "OU":
    action_noise = OrnsteinUhlenbeckNoise(mu, sigma, theta=args.theta, dt=args.dt)
  elif args.noise_type == "Gaussian":
    action_noise = GaussianNoise(mu, sigma)

  if args.noise_decay > 0:
    noise_decay  = PiecewiseSchedule([(0, 1.0), (args.noise_decay, 0.0)], outside_value=0.0)
    action_noise = DecayedExplorationNoise(action_noise, noise_decay)


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
  )

  ddpg_agent_kwargs = dict(
    model=model,
    model_kwargs=model_kwargs,
    actor_opt_conf=actor_opt_conf,
    critic_opt_conf=critic_opt_conf,
    action_noise=action_noise,
    update_target_freq=args.update_freq,
    memory_size=args.memory_size,
    obs_len=1,
  )

  kwargs = {**ddpg_agent_kwargs, **agent_kwargs}

  # Log the parameters for model
  log_info = [("seed", args.seed), ("extra_info", args.extra_info)]
  log_info += kwargs.items()
  rltf_log.log_params(log_info, args)

  # Create the agent
  ddpg_agent = AgentDDPG(**kwargs)


def main():
  # Create the agent
  ddpg_agent, args = make_agent()

  # Build the agent and the TF graph
  ddpg_agent.build()

  # Train or eval the agent
  if args.mode == 'train':
    ddpg_agent.train()
  else:
    ddpg_agent.eval()

  # Close on exit
  ddpg_agent.close()


if __name__ == "__main__":
  main()
