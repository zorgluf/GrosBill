from nicegui import ui
import random
from utils.agents import Agent
from utils.files import load_model
from typing import List
from utils.env import GBEnv

@ui.refreshable
def _gui_generic_buttons(env: GBEnv, callback = None):
    ui.button("Next step", on_click=lambda: callback(None)).bind_visibility_from(env, "current_player", backward=lambda current_player: current_player == -1)
    ui.button("Finish game.", on_click=lambda: ui.navigate.to("/")).bind_visibility_from(env, "done")

def play_step(env: GBEnv, agents: List[Agent], pov_player: int, human_action = None, choose_best_action = True):

  done = False
  while not done:
    if env.current_player == -1:
       action = -1
    else:
        current_player = agents[env.current_player]
        if current_player.name == 'human':
            if human_action == None:
                env.render(callback=lambda a: play_step(env, agents, pov_player, a), pov_player = pov_player)
                _gui_generic_buttons.refresh(env, callback=lambda a: play_step(env, agents, pov_player, a))
                return
            else:
                action = human_action
                human_action = None
        else:
            action = current_player.choose_action(env, choose_best_action = choose_best_action, mask_invalid_actions = True)

    _, _, done, _ , info = env.step(action)
    if info['next_step_no_action']:
      env.render(callback=lambda a: play_step(env, agents, pov_player, a), pov_player = pov_player)
      _gui_generic_buttons.refresh(env, callback=lambda a: play_step(env, agents, pov_player, a))
      return
  
  env.render(pov_player = pov_player)

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
            agent_obj = Agent(f"{agent} {i}", ppo_model)
        agents.append(agent_obj)
    
    return agents

class PlayOptions:

    device = 'cpu'

@ui.page('/frouge')
def frouge_page():
    from environments.frouge.envs.frouge import FlammeRougeEnv

    agents_names = ['human', 'best_model1', 'best_model2', 'best_model3', 'best_model4']
    pov_player = agents_names.index('human')
    env = FlammeRougeEnv(player_names=agents_names)
    # set seed
    seed = random.randint(0,1000)
    env.reset(seed = seed)
    # load agents
    agents = load_agents(env, ['human', 'best_model', 'best_model', 'best_model', 'best_model'], options.device)
    # start gui
    env.nicegui_page()
    _gui_generic_buttons(env,)
    # play game
    play_step(env, agents, pov_player=pov_player)

if __name__ in {"__main__", "__mp_main__"}:
    options = PlayOptions()
    ui.link('Flamme Rouge', frouge_page)
    ui.toggle(["cpu","cuda"], value="cpu").bind_value(options, 'device')

    ui.run(title='GrosBill')