from rltf.cmdutils      import cmdargs
from rltf.envs          import wrap_deepmind_ddpg
from rltf.utils         import rltf_log
from rltf.utils         import maker


def parse_args():
  model_choices = ["DDPG"]
  return cmdargs.parse_args(model_choices)


def make_agent():

  # Parse the command line args
  agent_kwargs, args = parse_args()

  # Construct the model directory and configure loggers
  model_dir = maker.make_model_dir(args)

  # Log the program parameters
  rltf_log.log_params(agent_kwargs.items(), args)

  # Get the environment maker
  env_kwargs = {**agent_kwargs.pop("env_kwargs"), **dict(
    env_id=args.env_id,
    seed=args.seed,
    wrap=wrap_deepmind_ddpg,
  )}
  env_maker = maker.get_env_maker(**env_kwargs)

  agent_kwargs = {**agent_kwargs, **dict(
    env_maker=env_maker,
    model_dir=model_dir,
  )}

  # Create the agent
  agent_type  = agent_kwargs.pop("agent")
  ddpg_agent  = agent_type(**agent_kwargs)

  return ddpg_agent, args


def main():
  # Create the agent
  ddpg_agent, args = make_agent()

  # Build the agent and the TF graph
  ddpg_agent.build()

  # Train or eval the agent
  if args.mode == 'train':
    ddpg_agent.train()
  else:
    ddpg_agent.play()

  # Close on exit
  ddpg_agent.close()


if __name__ == "__main__":
  main()
