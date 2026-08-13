import os

import argparse
import time
import logging
import random

from sb3_contrib import MaskablePPO as PPO1
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from utils.callbacks import SelfPlayCallback
from utils.files import reset_logs, reset_models, load_model
from utils.register import get_environment
from utils.selfplay import selfplay_wrapper

import config

def main(args):

    model_dir = os.path.join(config.MODELDIR, args.env_name)

    try:
        os.makedirs(model_dir)
    except:
        pass
    if not args.keep_logs:
        reset_logs()
    if args.reset:
        reset_models(model_dir)

    logger = logging.getLogger(__name__)
    if args.debug:
        logger.setLevel(config.DEBUG)
    else:
        logger.setLevel(config.INFO)

    if args.log_name == None:
        log_name = args.env_name
    else:
        log_name = args.log_name

    if args.seed == 0:
        seed = random.randint(0,1000)
    else:
        seed = args.seed
    set_random_seed(seed)

    logger.info('Setting up the selfplay training environment opponents...')
    base_env = get_environment(args.env_name)
    if args.reset:
        #build base model
        load_model(base_env(), 'base.zip', args.device)
    # Using vec_env for parallel training do not work with selfplay_wrapper, due to not calling reset() function
    env = make_vec_env(selfplay_wrapper(base_env), n_envs=args.n_envs, 
                       env_kwargs=dict(opponent_type = args.opponent_type, logger = logger, device = args.device),
                       vec_env_cls=SubprocVecEnv)

    params = {'gamma':args.gamma
        , 'clip_range':args.clip_param
        , 'ent_coef':args.entcoeff
        , 'n_epochs':args.n_epochs
        , 'n_steps':args.n_steps
        , 'batch_size':args.batch_size
        , 'verbose':0
        , 'tensorboard_log':config.LOGDIR
        , 'device': args.device
    }
    if args.learning_rate is not None:
        params['learning_rate'] = args.learning_rate

    time.sleep(5) # allow time for the base model to be saved out when the environment is created

    if args.reset or not os.path.exists(os.path.join(model_dir, 'best_model.zip')):
        logger.info('Loading the base PPO agent to train...')
        model = PPO1.load(os.path.join(model_dir, 'base.zip'), env, **params)
    else:
        logger.info('Loading the best_model.zip PPO agent to continue training...')
        model = PPO1.load(os.path.join(model_dir, 'best_model.zip'), env, **params)

    #Callbacks
    logger.info('Setting up the selfplay evaluation environment opponents...')
    callback_args = {
        'eval_env': selfplay_wrapper(base_env)(opponent_type = args.opponent_type, logger = logger, device = args.device),
        # fixed baseline opponent (base.zip): progress metric independent of promotions
        'base_eval_env': selfplay_wrapper(base_env)(opponent_type = 'base', logger = logger, device = args.device),
        'best_model_save_path' : config.TMPMODELDIR,
        'log_path' : config.LOGDIR,
        'eval_freq' : args.eval_freq,
        'n_eval_episodes' : args.n_eval_episodes,
        'deterministic' : False,
        'render' : False,
        'verbose' : 0
    }
        
    # Evaluate the agent against previous versions
    eval_callback = SelfPlayCallback(args.opponent_type, args.threshold, args.env_name, logger, **callback_args)

    logger.info('Setup complete - commencing learning...\n')

    model.learn(total_timesteps=args.total_timesteps, callback=[eval_callback], reset_num_timesteps = False, tb_log_name=log_name, progress_bar=False)

    env.close()
    del env


def cli() -> None:
  """Handles argument extraction from CLI and passing to main().
  Note that a separate function is used rather than in __name__ == '__main__'
  to allow unit testing of cli().
  """
  # Setup argparse to show defaults on help
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)


  parser.add_argument("--reset", "-r", action = 'store_true', default = False
                , help="Start retraining the model from scratch")
  parser.add_argument("--keep_logs", "-kl", action = 'store_true', default = False
                , help="Keep all previous logs")
  parser.add_argument("--opponent_type", "-o", type = str, default = 'mostly_best'
              , help="best / mostly_best / random / base - the type of opponent to train against")
  parser.add_argument("--debug", "-d", action = 'store_true', default = False
              , help="Debug logging")
  parser.add_argument("--verbose", "-v", action = 'store_true', default = False
              , help="Show observation in debug output")
  parser.add_argument("--env_name", "-e", type = str, default = 'tictactoe'
              , help="Which gym environment to train in: frouge, stotten")
  parser.add_argument("--log_name", "-log", type = str, default = None
              , help="Name of the experiment in tensorboard")
  parser.add_argument("--seed", "-s",  type = int, default = 0
            , help="Random seed. If 0, random")
  
  parser.add_argument("--total_timesteps", "-ts",  type = int, default = 1e9
            , help="Total timesteps for the whole training. Keep this almost infinite value and stop manually the training.")
  parser.add_argument("--eval_freq", "-ef",  type = int, default = 10240
            , help="How many timesteps should each actor contribute before the agent is evaluated. Default value is fine for most games.")
  parser.add_argument("--n_eval_episodes", "-ne",  type = int, default = 100
            , help="How many episodes should each actor contirbute to the evaluation of the agent. Default value is fine for most games.")
  parser.add_argument("--threshold", "-t",  type = float, default = 0.5
            , help="What score/reward must the agent achieve during evaluation to 'beat' the previous version and generate a new best model. Choose carefully, depending on the scoring scale of the game.")
  parser.add_argument("--gamma", "-g",  type = float, default = 0.99
            , help="The value of gamma in PPO (0.99: long term reward, 0.95: short term reward)")
  parser.add_argument("--clip_param", "-c",  type = float, default = 0.2
            , help="The clip paramater in PPO (0.1: Very cautious updates, more stable but slower learning, 0.3: More aggressive updates, faster learning but less stable)")
  parser.add_argument("--entcoeff", "-ent",  type = float, default = 0.005
            , help="The entropy coefficient in PPO (0.0: No exploration pressure → fast convergence, but risk of local optima, 0.001: Slight exploration encouragement, 0.005–0.01: Balanced exploration, >0.02: Strong exploration → can hurt performance if too random. Dependant to action space size: the larger, lower the entropy.)")

  parser.add_argument("--learning_rate", "-lr",  type = float, default = None
            , help="Override the PPO learning rate (default: keep the value saved in the loaded model, 3e-4 for a fresh base). Lower values (1e-4) recommended for the transformer policy (stottentr).")

  parser.add_argument("--n_epochs", "-oe",  type = int, default = 5
            , help="The number of epoch to train the PPO agent per batch. Default value is fine for most games.")
  parser.add_argument("--n_envs", "-n_envs",  type = int, default = 1
            , help="The number of envs to run in parallel.")
  parser.add_argument("--n_steps", "-os",  type = int, default = 2048
            , help="The step size for the PPO optimiser. Depends on the average number of step inside the game. A good value is 100*avg(game_length).")
  parser.add_argument("--batch_size", "-ob",  type = int, default = 128
            , help="The minibatch size in the PPO optimiser. As much as your hardware can handle.")

  parser.add_argument("--device", "-dev",  type = str, default = "cpu"
            , help="The device to use")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()