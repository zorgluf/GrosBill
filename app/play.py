from nicegui import ui, app
import random
import os
from utils.agents import Agent
from utils.files import load_model
from utils.register import get_trajectory_path
from typing import List, Tuple
from utils.env import GBEnv
from dataclasses import dataclass

from imitation.data.types import Trajectory
from imitation.data import serialize
from numpy import ndarray
import numpy as np

@ui.refreshable
def _gui_generic_buttons(env: GBEnv, callback = None):
    ui.button("Next step", on_click=lambda: callback(None)).bind_visibility_from(env, "current_player", backward=lambda current_player: current_player == -1)
    with ui.dialog() as dialog, ui.card():
        ui.label().bind_visibility_from(env,"done").bind_text_from(env,"winner_player",lambda w: f"Player {env.player_names[env.winner_player]} win !" if env.winner_player != None else "Winner undetermined.")
        ui.button("Close game", on_click=lambda: ui.navigate.to("/")).bind_visibility_from(env, "done")
    if env.done == True:
        dialog.open()


def play_step(env: GBEnv, agents: List[Agent], pov_player: int, human_action = None, choose_best_action = True, suggest = False, moves: List = None):
    assert moves is not None, "moves must be specified"

    done = False
    while not done:
        if env.current_player == -1:
            action = -1
        else:
            current_player = agents[env.current_player]
            if current_player.name == 'human':
                if human_action == None:
                    env.render(
                        callback=lambda a: play_step(env, agents, pov_player, a, suggest=suggest, moves=moves),   
                        pov_player = pov_player,
                        suggested_action = agents[-1].choose_action(env, choose_best_action=True) if suggest else None
                    )
                    _gui_generic_buttons.refresh(env, callback=lambda a: play_step(env, agents, pov_player, a, suggest=suggest, moves=moves))
                    return
                else:
                    action = human_action
                    human_action = None
            else:
                action = current_player.choose_action(env, choose_best_action = choose_best_action)

        obs, _, done, _ , info = env.step(action)
        if (type(action) != int or action != -1) and current_player.name == 'human':
            #record for trajectory
            moves[0].append(obs)
            moves[1].append(action)
        if info['next_step_no_action']:
            env.render(callback=lambda a: play_step(env, agents, pov_player, a, moves=moves), pov_player = pov_player, suggest=suggest, moves=moves)
            _gui_generic_buttons.refresh(env, callback=lambda a: play_step(env, agents, pov_player, a, moves=moves))
            return
  
    env.render(pov_player = pov_player)
    _gui_generic_buttons.refresh(env)
    if app.storage.user["options"].record:
        save_trajectory(moves[0], moves[1], env.name)

def save_trajectory(observations, actions, env_name):
    #save trajectory
    traj = Trajectory(obs=observations, acts=actions, infos=None, terminal=True)
    if os.path.exists(get_trajectory_path(env_name)):
        trajectories = serialize.load(get_trajectory_path(env_name))
    else:
        trajectories = []
    trajectories.append(traj)
    serialize.save(get_trajectory_path(env_name), trajectories)

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

    if app.storage.user["options"].suggest:
        # load best agent for suggestion
        ppo_model = load_model(env, f'best_model.zip', device)
        agent_obj = Agent(f"Suggest Agent", ppo_model)
        agents.append(agent_obj)
    
    return agents

@dataclass
class PlayOptions:
    suggest = False
    record = False

@ui.page('/frouge')
def frouge_page():
    from environments.frouge.envs.frouge import FlammeRougeEnv

    agents_names = ['human', 'best_model1', 'best_model2', 'best_model3', 'best_model4']
    pov_player = agents_names.index('human')
    env = FlammeRougeEnv(player_names=agents_names)
    # set seed
    seed = random.randint(0,1000)
    obs, _ = env.reset(seed = seed)
    # load agents
    agents = load_agents(env, ['human', 'best_model', 'best_model', 'best_model', 'best_model'], "cpu")
    # start gui
    env.nicegui_page()
    _gui_generic_buttons(env,)
    # play game
    play_step(env, agents, pov_player=pov_player, suggest=app.storage.user["options"].suggest, moves=[[obs],[]])

@ui.page('/stotten')
def stotten_page():
    #TODO : refactor this to use the same code as frouge_page
    from environments.stotten.envs.stotten import SchottenTottenEnv

    agents_names = ['human', 'computer']
    pov_player = agents_names.index('human')
    env = SchottenTottenEnv(player_names=agents_names)
    # set seed
    seed = random.randint(0,1000)
    obs, _ = env.reset(seed = seed)
    # load agents
    agents = load_agents(env, ['human', 'best_model'], "cpu")
    # start gui
    env.nicegui_page()
    _gui_generic_buttons(env,)
    # play game
    play_step(env, agents, pov_player=pov_player, suggest=app.storage.user["options"].suggest, moves=[[obs],[]])


@ui.page('/')
def index():
    #init options on user scope
    app.storage.user["options"] = PlayOptions()

    ui.link('Flamme Rouge', frouge_page)
    ui.link('Schotten Totten', stotten_page)
    with ui.row():
        ui.label('Suggest action:')
        ui.toggle({True:"Yes",False:"No"}).bind_value(app.storage.user["options"], 'suggest')
        ui.toggle({True:"Record for future learning",False:"No"}).bind_value(app.storage.user["options"], 'record')

if __name__ in {"__main__", "__mp_main__"}:

    ui.run(title='GrosBill', storage_secret='private key almost impossible to guess')