import argparse
import datetime
import glob
import gym
import numpy as np
import os
import random
import tensorflow as tf

from rltf.agents        import AgentDDPG
from rltf.env_wrappers  import wrap_deepmind_ddpg
from rltf.exploration   import OrnsteinUhlenbeckNoise
from rltf.models        import DDPG
from rltf.optimizers    import OptimizerConf
from rltf.optimizers    import AdamGradClipOptimizer
from rltf.schedules     import ConstSchedule


def set_global_seeds(i):
  tf.set_random_seed(i) 
  np.random.seed(i)
  random.seed(i)


def make_env(env_id, seed, model_dir, no_video, video_freq=None):

  # Set all seeds
  set_global_seeds(seed)

  env_file = model_dir + "Env.pkl"
  if os.path.isfile(env_file):
    return pickle_restore(env_file)

  gym_dir = model_dir + "gym_video"
  if no_video:
    video_callable = lambda e_id: False
  else:
    if video_freq is None:
      video_callable = None
    else:
      video_callable = lambda e_id: e_id % video_freq == 0

  env = gym.make(env_id)
  env.seed(seed)
  env = gym.wrappers.Monitor(env, gym_dir, force=True, video_callable=video_callable)

  return env


def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--env-id',       required=True,  type=str,   help='full environment name')
  # parser.add_argument('--model',        required=True,  type=str,   help='network model',
  #                     default="DDPG",   choices=["DDPG", "QRDDPG"])

  parser.add_argument('--actor-lr',     default=1e-4,   type=float, help='actor learn rate',)
  parser.add_argument('--critic-lr',    default=1e-3,   type=float, help='critic learn rate')
  parser.add_argument('--tau',          default=0.001,  type=float, help='target soft update weight')
  parser.add_argument('--sigma',        default=0.2,    type=float, help='ornsein uhlenbeck sigma')
  parser.add_argument('--theta',        default=0.15,   type=float, help='ornsein uhlenbeck theta')
  parser.add_argument('--critic_reg',   default=0.02,   type=float, help='network weight regularization')
  
  parser.add_argument('--adam-epsilon', default=1e-4,   type=float, help='expsilon for Adam optimizer')
  parser.add_argument('--train-freq',   default=1,      type=int,   help='learn frequency')
  parser.add_argument('--seed',         default=0,      type=int,   help='seed')
  parser.add_argument('--grad-clip',    default=0,      type=float, 
                      help='value to clip gradinets to. If 0, no clipping')
  parser.add_argument('--extra-info',   default="",     type=str,   help='extra info to log')

  parser.add_argument('--video-freq',   default=500,    type=int,
                      help='frequency in number of episodes at whcih to record videos')
  parser.add_argument('--huber-loss',   help='use huber_loss',    action="store_true")
  parser.add_argument('--save',         help='save model',        action="store_true")
  parser.add_argument('--no-video',     help='save gym videos',   action="store_true")

  
  args = parser.parse_args()
  
  if args.grad_clip > 0 and args.huber_loss:
    raise ValueError("Cannot use huber loss and gradient clipping simultaneously")
  # assert not (args.huber_loss and args.grad_clip > 0)

  return args



def make_model_dir(model_type, env_id):
  model_name  = model_type.__name__.lower()
  project_dir = os.path.dirname(os.path.abspath(__file__))
  model_dir   = os.path.join(project_dir, "trained_models")
  model_dir   = os.path.join(model_dir,   model_name)
  model_dir   = os.path.join(model_dir,   env_id)

  # Get the number of existing models
  pattern     = model_dir + "_m*/"
  models      = glob.glob(pattern)
  
  # Get the number of the new model dir
  model_dir  += "_m" + str(len(models)+1)
  model_dir   = os.path.join(model_dir, "")

  # Create the directory for the model
  os.makedirs(model_dir)
  
  return model_dir


def log_params(model_dir, params):
  params = sorted(params, key=lambda tup: tup[0])

  str_sizes = [len(s) for s, _ in params]
  pad       = max(str_sizes) + 2
  params    = [(s.ljust(pad), v) for s, v in params]

  with open(os.path.join(model_dir, "params.txt"), 'w') as f:
    date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f.write(date + "\n\n")
    for k, v in params:
      f.write(k + ": " + str(v) + "\n")


def main():

  args = parse_args()

  model_type = DDPG

  # Get the model directory path and save the arguments for the run
  model_dir = make_model_dir(model_type, args.env_id)

  # Set learning rates and optimizer configuration
  actor_lr  = ConstSchedule(args.actor_lr)
  critic_lr = ConstSchedule(args.critic_lr)

  if args.grad_clip > 0.0:
    opt_args = dict(epsilon=args.adam_epsilon, grad_clip=args.grad_clip)
    actor_opt_conf  = OptimizerConf(AdamGradClipOptimizer, actor_lr,  **opt_args)
    critic_opt_conf = OptimizerConf(AdamGradClipOptimizer, critic_lr, **opt_args)
  else:
    actor_opt_conf  = OptimizerConf(tf.train.AdamOptimizer, actor_lr,  epsilon=args.adam_epsilon)
    critic_opt_conf = OptimizerConf(tf.train.AdamOptimizer, critic_lr, epsilon=args.adam_epsilon)

  # Create the exploration noise
  action_noise = OrnsteinUhlenbeckNoise(mu=0, sigma=args.sigma, theta=args.theta)


  # Create the environment
  env = make_env(args.env_id, args.seed, model_dir, args.no_video, args.video_freq)
  env = wrap_deepmind_ddpg(env)

  # Set additional arguments
  batch_size = 16 if len(env.observation_space.shape) == 3 else 64

  # Set the Agent class keyword arguments
  agent_config = dict(
    env=env,
    train_freq=args.train_freq,
    start_train=1000,
    max_steps=int(2.5e6),
    batch_size=batch_size,
    model_dir=model_dir,
    save=args.save,
  )

  # Set the model-specific keyword arguments
  model_kwargs = dict(
    critic_reg=args.critic_reg,
    tau=args.tau,
    gamma=0.99,
    huber_loss=args.huber_loss,
  )

  # Log the parameters for model
  log_info = [
    ("actor_opt_conf",  actor_opt_conf),
    ("critic_opt_conf", critic_opt_conf),
    ("action_noise",    action_noise),
    ("seed",            args.seed),
    ("extra_info",      args.extra_info),
  ]
  log_info += agent_config.items()
  log_info += model_kwargs.items()

  log_params(model_dir, log_info)

  # Create the agent
  ddpg_agent = AgentDDPG(
    agent_config=agent_config, 
    model_type=model_type, 
    model_kwargs=model_kwargs,
    actor_opt_conf=actor_opt_conf,
    critic_opt_conf=critic_opt_conf,
    action_noise=action_noise,
  )

  # Build the agent and the TF graph
  ddpg_agent.build()

  # Train the agent
  ddpg_agent.train()

  # Close on exit
  ddpg_agent.close()
  env.close()

if __name__ == "__main__":
  main()
