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

players = list()
choose_best_action = False

def play_step(env, human_action = None):

  if human_action == -1:
    #close game
    env.close()
    return

  done = False
  while not done:

    current_player = players[env.current_player_num]
    if current_player.name == 'human':
      if human_action == None:
        env.render(callback=lambda a: play_step(env, a))
        return
      else:
        action = human_action
        human_action = None
    elif current_player.name == 'rules':
      action = current_player.choose_action(env, choose_best_action = False, mask_invalid_actions = True)
    else:
      action = current_player.choose_action(env, choose_best_action = choose_best_action, mask_invalid_actions = True)

    obs, reward, done, _ , _ = env.step(action)
  
  env.render(callback=lambda a: play_step(env, a))

def main(args):

  logger = configure(config.LOGDIR)

  render_mode = "human_web"

  if args.seed == 0:
    seed = random.randint(0,1000)
  else:
    seed = args.seed
    
  #make environment
  env = get_environment(args.env_name)(verbose = False, manual = False, render_mode = render_mode)
  set_random_seed(seed)

  global choose_best_action
  choose_best_action = args.best

  agents = []

  #load the agents
  if len(args.agents) != env.n_players:
    raise Exception(f'{len(args.agents)} players specified but this is a {env.n_players} player game!')

  for i, agent in enumerate(args.agents):
    if agent == 'human':
      agent_obj = Agent('human')
    elif agent == 'rules':
      agent_obj = Agent('rules')
    elif agent == 'base':
      base_model = load_model(env, 'base.zip', args.device)
      agent_obj = Agent('base', base_model)   
    else:
      ppo_model = load_model(env, f'{agent}.zip', args.device)
      agent_obj = Agent(agent, ppo_model)
    agents.append(agent_obj)
  
  global players
  players = agents[:]

  if args.randomise_players:
    random.shuffle(players)

  obs = env.reset(seed = seed)

  #start the game
  play_step(env)


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
  parser.add_argument("--randomise_players", "-r",  action = 'store_true', default = False
            , help="Randomise the player order")
  parser.add_argument("--cont", "-c",  action = 'store_true', default = False
            , help="Pause after each turn to wait for user to continue")
  parser.add_argument("--env_name", "-e",  type = str, default = 'TicTacToe'
            , help="Which game to play?")
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