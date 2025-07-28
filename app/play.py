from nicegui import ui
import random
from utils.agents import Agent
from utils.files import load_model
from typing import List
from utils.env import GBEnv

def play_step(env: GBEnv, agents: List[Agent], human_action = None, choose_best_action = True):

  if human_action == -1:
    #close game
    env.close()
    return

  done = False
  while not done:

    current_player = agents[env.current_player]
    if current_player.name == 'human':
      if human_action == None:
        env.render(callback=lambda a: play_step(env, agents, a), pov_player = env.current_player)
        return
      else:
        action = human_action
        human_action = None
    else:
      action = current_player.choose_action(env, choose_best_action = choose_best_action, mask_invalid_actions = True)

    _, _, done, _ , _ = env.step(action)
  
  env.render(callback=lambda a: play_step(env, agents, a), pov_player = env.current_player)

def load_agents(env, agent_names, device):

    if len(agent_names) != env.n_players:
        raise Exception(f'{len(agent_names)} players specified but this is a {env.n_players} player game!')
    agents = []
    for i, agent in enumerate(agent_names):
        if agent == 'human':
            agent_obj = Agent('human')
        elif agent == 'base':
            base_model = load_model(env, 'base.zip', device)
            agent_obj = Agent('base', base_model)   
        else:
            ppo_model = load_model(env, f'{agent}.zip', device)
            agent_obj = Agent(agent, ppo_model)
        agents.append(agent_obj)
    
    return agents

class PlayOptions:

    device = 'cpu'

@ui.page('/frouge')
def frouge_page():
    from environments.frouge.envs.frouge import FlammeRougeEnv

    env = FlammeRougeEnv()
    # set seed
    seed = random.randint(0,1000)
    env.reset(seed = seed)
    # load agents
    agents = load_agents(env, ['human', 'best_model', 'best_model', 'best_model', 'best_model'], options.device)
    # start gui
    env.nicegui_page()
    # play game
    play_step(env, agents)

if __name__ in {"__main__", "__mp_main__"}:
    options = PlayOptions()
    ui.link('Flamme Rouge', frouge_page)
    ui.toggle(["cpu","cuda"], value="cpu").bind_value(options, 'device')

    ui.run(title='GrosBill')