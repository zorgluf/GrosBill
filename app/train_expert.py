'''
Contexte:
- This python script train a PPO agent to play boardgame described in a gym environment.
Instructions:
- Change this python script to introduce a mixed training : learning from expert records, then perfect the training with PPO as done in the original script.
- be creative and suggest some enhancement if it's usefull for the training.
Usefull information:
- The expert records will be passed as argument in the command line. The records will be stored in a list of trajectories, using the python lib "imitation".
- The export records are just records from a standard human player, used for the agent to have a good bootstrap of the way to play, do not take them as the ultimate best moves.
- The PPO uses a custom MLP policy, dependant on the game, and located in models directory.
'''
import os

import argparse
import time
import logging
import random
import numpy as np

from sb3_contrib import MaskablePPO as PPO1
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed, FloatSchedule

from imitation.data import rollout
from imitation.algorithms import bc
from imitation.data.types import map_maybe_dict, maybe_wrap_in_dictobs, Trajectory, DictObs

from utils.callbacks import SelfPlayCallback
from utils.files import reset_logs, reset_models, load_model
from utils.register import get_environment, get_network_arch, get_trajectory_path
from utils.selfplay import selfplay_wrapper
from utils.experts import load_trajectories

import config

def load_expert_trajectories(env_name, env, logger):

    # Load trajectories from file
    trajectories = load_trajectories(env_name)
    logger.info(f"Loaded {len(trajectories)} expert trajectories")
    # convert to dictobs for imitation library
    rewrap_trajectories = []
    for traj in trajectories:
        obs = DictObs.stack([ maybe_wrap_in_dictobs(map_maybe_dict(np.array, obs)) for obs in traj.obs ])
        rewrap_trajectories.append(Trajectory(obs=obs, acts=np.array(traj.acts), infos=traj.infos, terminal=traj.terminal))

    # Convert trajectories to the format expected by BC
    transitions = rollout.flatten_trajectories(rewrap_trajectories)
    logger.info(f"Processed {len(transitions)} expert transitions")

    return transitions


def train_behavioral_cloning(expert_transitions, env, params, logger, seed):
    """Train a policy using behavioral cloning on expert demonstrations."""
    if expert_transitions is None or len(expert_transitions) == 0:
        logger.warning("No expert transitions available, skipping behavioral cloning")
        return None
    
    try:
        # Get the custom policy class for this environment
        policy_class = params.get('policy_class')
        lr = params.get('lr', 3e-4)
        
        # Create BC trainer with custom policy
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=expert_transitions,
            rng=np.random.default_rng(seed),
            policy=policy_class(env.observation_space, env.action_space, lr_schedule=FloatSchedule(lr)),
            batch_size=min(32, len(expert_transitions))  # Adjust batch size based on available transitions
        )
        
        logger.info("Starting behavioral cloning training...")
        bc_trainer.train(n_epochs=params.get('bc_epochs', 50))
        
        # Get the trained policy
        bc_policy = bc_trainer.policy
        logger.info("Behavioral cloning training completed")
        
        return bc_policy
    except Exception as e:
        logger.error(f"Behavioral cloning failed: {e}")
        return None


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
    policy_class = get_network_arch(args.env_name)
    if args.reset:
        #build base model
        load_model(base_env(), 'base.zip', args.device)
    # Using vec_env for parallel training do not work with selfplay_wrapper, due to not calling reset() function
    env = make_vec_env(selfplay_wrapper(base_env), n_envs=args.n_envs, 
                       env_kwargs=dict(opponent_type = args.opponent_type, logger = logger, device = args.device),
                       vec_env_cls=SubprocVecEnv)
    #env = selfplay_wrapper(base_env)(opponent_type = args.opponent_type, logger = logger, device = args.device)

    params = {'gamma':args.gamma
        , 'clip_range':args.clip_param
        , 'ent_coeff':args.entcoeff
        , 'n_epochs':args.n_epochs
        , 'n_steps':args.n_steps
        , 'batch_size':args.batch_size
        , 'verbose':0
        , 'tensorboard_log':config.LOGDIR
        , 'device_str': str(args.device)
        , 'bc_epochs': args.bc_epochs
        , 'lr': args.lr
        , 'policy_class': policy_class
    }

    time.sleep(5) # allow time for the base model to be saved out when the environment is created

    # Load expert trajectories if available
    expert_trajectories = None
    logger.info(f"Loading expert trajectories from {get_trajectory_path(args.env_name)}...")
    expert_trajectories = load_expert_trajectories(args.env_name, env, logger)

    # Behavioral Cloning phase
    bc_policy = None
    if expert_trajectories is not None and len(expert_trajectories) > 0:
        logger.info("Starting behavioral cloning phase...")
        bc_policy = train_behavioral_cloning(expert_trajectories, env, params, logger, seed)

    # Initialize PPO model
    if args.reset or not os.path.exists(os.path.join(model_dir, 'best_model.zip')):
        logger.info('Loading the base PPO agent to train...')
        model = PPO1.load(os.path.join(model_dir, 'base.zip'), env, **params)
        
        # If we have a BC policy, use it to warm start the PPO model
        if bc_policy is not None:
            logger.info("Warm starting PPO model with behavioral cloning policy...")
            model.policy.load_state_dict(bc_policy.state_dict())
    else:
        logger.info('Loading the best_model.zip PPO agent to continue training...')
        model = PPO1.load(os.path.join(model_dir, 'best_model.zip'), env, **params)

    #Callbacks
    logger.info('Setting up the selfplay evaluation environment opponents...')
    callback_args = {
        'eval_env': selfplay_wrapper(base_env)(opponent_type = args.opponent_type, logger = logger, device = args.device),
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
  parser.add_argument("--entcoeff", "-ent",  type = float, default = 0.05
            , help="The entropy coefficient in PPO (0.0: No exploration pressure → fast convergence, but risk of local optima, 0.01: Slight exploration encouragement, 0.05–0.1: Balanced exploration, >0.2: Strong exploration → can hurt performance if too random.)")

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

  parser.add_argument("--lr", "-lr", type = float, default = 3e-4
            , help="Learning rate for the policy optimizer")

  parser.add_argument("--bc_epochs", "-bce", type = int, default = 50
            , help="Number of epochs for behavioral cloning training")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()