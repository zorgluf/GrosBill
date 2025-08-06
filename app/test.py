# docker-compose exec app python3 test.py -d -g 1 -a base base human -e butterfly 

import logging
import random
import argparse

from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

from utils.files import load_model, write_results
from utils.register import get_environment
from utils.agents import Agent

import config
import numpy as np


def main(args):
    # Set up logging
    logger = logging.getLogger("test.py")
    if args.debug:
        logger.setLevel(config.DEBUG)
    else:
        logger.setLevel(config.INFO)

    # Set up seed
    if args.seed == 0:
        seed = random.randint(0,1000)
    else:
        seed = args.seed

    #make environment
    env = get_environment(args.env_name)()
    set_random_seed(seed)

    total_rewards = {}

    agents = []

    #load the agents
    if len(args.agents) != env.n_players:
        raise Exception(f'{len(args.agents)} players specified but this is a {env.n_players} player game!')

    for i, agent in enumerate(args.agents):
        if agent == 'base':
            base_model = load_model(env, 'base.zip', args.device)
            agent_obj = Agent('base', base_model)   
        else:
            ppo_model = load_model(env, f'{agent}.zip', args.device)
            agent_obj = Agent(agent, ppo_model)
        agents.append(agent_obj)
        total_rewards[agent_obj.id] = 0
  
    #play games
    logger.info(f'Playing {args.games} games...')
    for game in range(args.games):
        players = agents[:]

        if args.randomise_players:
            random.shuffle(players)

        obs = env.reset(seed = seed)
        done = False
    
        for i, p in enumerate(players):
            logger.debug(f'Player {i+1} = {p.name}')

        while not done:

            current_player = players[env.current_player]

            action = current_player.choose_action(env, choose_best_action = args.best, mask_invalid_actions = True)
            logger.debug(f'Current player name: {current_player.name}, choosing action: {action}')

            obs, reward, done, _ , info = env.step(action)
            while info['next_step_no_action'] and not done:
                obs, reward, done, _ , info = env.step(-1)
                logger.debug(f'No action needed, continuing...')

            for r, player in zip(reward, players):
                total_rewards[player.id] += r
                player.points += r


        logger.info(f"Played {game + 1} games: {total_rewards}")

        if args.write_results:
            write_results(players, game, args.games, env.turns_taken)

        for p in players:
            p.points = 0

        env.close()


def cli() -> None:
  """Handles argument extraction from CLI and passing to main().
  Note that a separate function is used rather than in __name__ == '__main__'
  to allow unit testing of cli().
  """
  # Setup argparse to show defaults on help
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  parser.add_argument("--agents","-a", nargs = '+', type=str, default = ['human', 'human']
                , help="Player Agents (human, ppo version)")
  parser.add_argument("--best", "-b", action = 'store_true', default = False
                , help="Make AI agents choose the best move (rather than sampling)")
  parser.add_argument("--games", "-g", type = int, default = 1
                , help="Number of games to play)")
  parser.add_argument("--debug", "-d",  action = 'store_true', default = False
            , help="Show logs to debug level")
  parser.add_argument("--verbose", "-v",  action = 'store_true', default = False
            , help="Show observation on debug logging")
  parser.add_argument("--randomise_players", "-r",  action = 'store_true', default = False
            , help="Randomise the player order")
  parser.add_argument("--env_name", "-e",  type = str, default = 'TicTacToe'
            , help="Which game to play?")
  parser.add_argument("--write_results", "-w",  action = 'store_true', default = False
            , help="Write results to a file?")
  parser.add_argument("--seed", "-s",  type = int, default = 0
            , help="Random seed. If 0, random")
  
  parser.add_argument("--device", "-dev",  type = str, default = "cpu"
            , help="The device to use")
  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()