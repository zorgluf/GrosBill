import os

import argparse
import time
import logging
import random
import gymnasium as gym
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor, set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils.callbacks import SelfPlayCallback
from utils.files import reset_logs, reset_models, load_model
from utils.register import get_environment
from utils.selfplay import selfplay_wrapper

import config

from gymcts.gymcts_neural_agent import GymctsNeuralAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymcts.logger import log

def generate_trajectories(agent: GymctsNeuralAgent, logger = logging.getLogger(__name__)):
    states, actions, action_masks, rewards, dones = [], [], [], [], []
    done = False
    state = agent.search_root_node._obs
    next_node = agent.search_root_node
    while not done:
        # MCTS choisit une action
        action_masks.append(next_node.state.env.env.action_masks())
        action, next_node = agent.perform_mcts_step()
        # Enregistre l'état et l'action
        states.append(state)
        actions.append(action)
        reward = next_node.state._step_tuple[1]
        rewards.append(reward)
        dones.append(False)
        done = next_node.terminal
        state = next_node._obs
    if dones:  # Marque la dernière step comme done
        dones[-1] = True
    return states, actions, action_masks, rewards, dones

def train_offline_ppo(model, states, actions, action_masks, rewards, dones, args, env, logger, callbacks):
    """Entraîne PPO offline sur des trajectoires pré-collectées sans rollouts."""

    # Crée un buffer dédié pour l'entraînement offline
    if isinstance(env.observation_space, gym.spaces.Dict):
        offline_buffer = MaskableDictRolloutBuffer(
            buffer_size=len(states),
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=model.device,
            gamma=args.gamma,
            gae_lambda=0.95
        )
        states = [ { k:np.expand_dims(np.array(v),axis=0) for k,v in state.items()} for state in states ]
    else:
        offline_buffer = MaskableRolloutBuffer(
            buffer_size=len(states),
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=model.device,
            gamma=args.gamma,
            gae_lambda=0.95
        )

    # Remplit le buffer
    offline_buffer.reset()
    model.policy.set_training_mode(False)
    num_timesteps = len(states)
    for i, obs in enumerate(states):
        # Prédit les valeurs et log_probs avec le modèle actuel (old_log_probs)
        with torch.no_grad():
            obs_tensor = obs_as_tensor(obs, model.device)
            _, value, log_prob = model.policy(obs_tensor, action_masks=action_masks[i])
        offline_buffer.add(
            obs=obs,
            action=np.array([actions[i]]),
            reward=rewards[i],
            episode_start=dones[i],
            value=value,
            log_prob=log_prob,
            action_masks=action_masks[i]
        )
        offline_buffer.compute_returns_and_advantage(last_values=value, dones=np.array([dones[i]]))
    model.policy.set_training_mode(True)

    # Échange temporairement le buffer du modèle
    original_buffer = model.rollout_buffer
    model.rollout_buffer = offline_buffer

    # Entraîne sur ces données sans rollouts
    _, callback = model._setup_learn(
        1e10,
        callbacks,
        False,
        "MCTSlearn",
        False,
    )
    callback.on_training_start(locals(), globals())
    for i in range(args.n_epochs):
        logger.info(f"Train on MCTS data (epoch {i}/{args.n_epochs})")
        model.train()
        model.num_timesteps += num_timesteps
        model.dump_logs(i)
        for _ in range(num_timesteps):
            callback.on_step()
    callback.on_training_end()
    model.policy.set_training_mode(False)

    # Restaure le buffer original
    model.rollout_buffer = original_buffer

def main(args):
   
    log.setLevel(20)
    #log.setLevel(10) #for debug

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
        load_model(get_environment(args.env_name)(), 'base.zip', args.device)

    # Initialisation
    env_self = selfplay_wrapper(base_env)(opponent_type = args.opponent_type, logger = logger, device = args.device)
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

    time.sleep(5) # allow time for the base model to be saved out when the environment is created

    if args.reset or not os.path.exists(os.path.join(model_dir, 'best_model.zip')):
        logger.info('Loading the base PPO agent to train...')
        model = MaskablePPO.load(os.path.join(model_dir, 'base.zip'), env_self, **params)
    else:
        logger.info('Loading the best_model.zip PPO agent to continue training...')
        model = MaskablePPO.load(os.path.join(model_dir, 'best_model.zip'), env_self, **params)
    env_self.reset()

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
    eval_callback = SelfPlayCallback(args.opponent_type, args.threshold, args.env_name, logger, **callback_args)

    # Boucle d'amélioration
    for iteration in range(args.nb_improve_loop): 
        logger.info(f"Iteration {iteration + 1}/{args.nb_improve_loop}")

        # 1. Créez l'agent MCTS avec la politique SB3 actuelle et générer des trajectoires
        states, actions, action_masks, rewards, dones = [], [], [], [], []
        for i in range(args.nb_episode_gen):
            logger.info(f"Generate trajectories {i}/{args.nb_episode_gen}")
            env_self.reset()
            env = DeepCopyMCTSGymEnvWrapper(env_self, action_mask_fn=lambda env: env.env.action_masks())
            agent = GymctsNeuralAgent(
                env=env,
                render_tree_after_step=False,
                render_tree_max_depth=3,
                exclude_unvisited_nodes_from_render=False,
                number_of_simulations_per_step=args.nb_sim_mcts,
                clear_mcts_tree_after_step = True,
                model=model
            )
            logger.info(f"Generate trajectories")
            ep_states, ep_actions, ep_action_masks, ep_rewards, ep_dones = generate_trajectories(agent, logger=logger)
            states += ep_states
            actions += ep_actions
            action_masks += ep_action_masks
            rewards += ep_rewards
            dones += ep_dones

        # Entraîne le modèle PPO uniquement sur les trajectoires MCTS générées
        if states:
            train_offline_ppo(model, states, actions, action_masks, rewards, dones, args, env_self, logger, callbacks=[eval_callback])

        # 3. Réentraînez le modèle SB3 en mode standard et standalone
        #logger.info(f"Train policy on standard on-policy rollout")
        #model.learn(total_timesteps=args.total_timesteps, callback=[eval_callback], reset_num_timesteps = False, tb_log_name=log_name, progress_bar=False)

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
  
  parser.add_argument("--total_timesteps", "-ts",  type = int, default = 1e4
            , help="Total timesteps for the whole training. Keep this almost infinite value and stop manually the training.")
  parser.add_argument("--eval_freq", "-ef",  type = int, default = 10240
            , help="How many timesteps should each actor contribute before the agent is evaluated. Default value is fine for most games.")
  parser.add_argument("--n_eval_episodes", "-ne",  type = int, default = 100
            , help="How many episodes should each actor contirbute to the evaluation of the agent. Default value is fine for most games.")
  parser.add_argument("--threshold", "-t",  type = float, default = 0.85
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

  parser.add_argument("--nb_episode_gen", "-negen",  type = int, default = 100
            , help="The nubmer of episode to generate with mcts explorations.")
  parser.add_argument("--nb_improve_loop", "-niloop",  type = int, default = 10
            , help="The number of improvement loop (mcts search + PPO training of model).")
  parser.add_argument("--nb_sim_mcts", "-simmcts",  type = int, default = 10
            , help="The number of MCTS simulations per step.")

  parser.add_argument("--device", "-dev",  type = str, default = "cpu"
            , help="The device to use")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()


