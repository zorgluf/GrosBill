import subprocess
import argparse
import logging
import os
import itertools

import config
from utils.files import reset_logs, reset_models

"""
Simple script to test different hyperparameter values in order to select the best one.
Results available through tensorboard
"""
HYPER_PARAMS = dict(
    gamma = [ 0.99 ],
    clip = [ 0.1, 0.2, 0.3 ],
    ent = [ 0, 0.01, 0.05, 0.1, 0.2 ],
    epoch = [ 2, 5 ],
    batch = [ 128 ]
)

def main(args):

    model_dir = os.path.join(config.MODELDIR, args.env_name)

    try:
        os.makedirs(model_dir)
    except:
        pass
    reset_logs()
    reset_models(model_dir)

    logger = logging.getLogger("train_hyper")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(config.INFO)

    logger.info(f'Testing the following hyperparameters on env {args.env_name}')
    logger.info(HYPER_PARAMS)
    #Get all permutation of hyperparameters values
    params, values = zip(*HYPER_PARAMS.items())
    permutations_params = [dict(zip(params, v)) for v in itertools.product(*values)]
    for params in permutations_params:
       logger.info(f"Launch training with parameters: {params}")
       subprocess.run([ "python", "train.py", "-e", args.env_name, "-r",
                       "-g", str(params["gamma"]),
                       "-c", str(params["clip"]),
                       "-ent", str(params["ent"]),
                       "-oe", str(params["epoch"]),
                       "-ob", str(params["batch"]),
                       "-kl", "-log", str(params),
                       "-dev", args.device,
                       "-n_envs", str(args.n_envs),
                       "-t", str(args.threshold),
                       "-ts", str(args.total_timesteps)])

    logger.info('Trainings finished.')



def cli() -> None:
  """Handles argument extraction from CLI and passing to main().
  Note that a separate function is used rather than in __name__ == '__main__'
  to allow unit testing of cli().
  """
  # Setup argparse to show defaults on help
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  parser.add_argument("--env_name", "-e", type = str, default = 'tictactoe'
              , help="Which gym environment to train in: frouge, stotten")
  
  parser.add_argument("--threshold", "-t",  type = float, default = 0.5
            , help="What score/reward must the agent achieve during evaluation to 'beat' the previous version and generate a new best model. Choose carefully, depending on the scoring scale of the game.")

  parser.add_argument("--n_envs", "-n_envs",  type = int, default = 1
            , help="The number of envs to run in parallel.")

  parser.add_argument("--device", "-dev",  type = str, default = "cpu"
            , help="The device to use")
  parser.add_argument("--total_timesteps", "-ts",  type = int, default = 500000
            , help="Total timesteps for the whole training. Set this value corresponding your compute power.")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()