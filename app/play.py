from nicegui import ui, app
import random
import os
import shutil
import config
from utils.agents import Agent
from utils.files import load_model
from typing import List, Tuple
from utils.env import GBEnv
from dataclasses import dataclass

@ui.refreshable
def _gui_generic_buttons(env: GBEnv, callback = None):
    ui.button("Next step", on_click=lambda: callback(None)).bind_visibility_from(env, "current_player", backward=lambda current_player: current_player == -1)
    with ui.dialog() as dialog, ui.card():
        ui.label().bind_visibility_from(env,"done").bind_text_from(env,"winner_player",lambda w: f"Player {env.player_names[env.winner_player]} win !" if env.winner_player != None else "Winner undetermined.")
        ui.button("Close game", on_click=lambda: ui.navigate.to("/")).bind_visibility_from(env, "done")
    if env.done == True:
        dialog.open()


def play_step(env: GBEnv, agents: List[Agent], pov_player: int, human_action = None, choose_best_action = True, suggest = False):
    done = False
    while not done:
        if env.current_player == -1:
            action = -1
        else:
            current_player = agents[env.current_player]
            if current_player.name == 'human':
                if human_action == None:
                    env.render(
                        callback=lambda a: play_step(env, agents, pov_player, a, suggest=suggest),   
                        pov_player = pov_player,
                        suggested_action = agents[-1].choose_action(env, choose_best_action=True) if suggest else None
                    )
                    _gui_generic_buttons.refresh(env, callback=lambda a: play_step(env, agents, pov_player, a, suggest=suggest))
                    return
                else:
                    action = human_action
                    human_action = None
            else:
                action = current_player.choose_action(env, choose_best_action = choose_best_action)

        obs, _, done, _ , info = env.step(action)
        if info['next_step_no_action']:
            env.render(callback=lambda a: play_step(env, agents, pov_player, a), pov_player = pov_player, suggest=suggest)
            _gui_generic_buttons.refresh(env, callback=lambda a: play_step(env, agents, pov_player, a))
            return
  
    env.render(pov_player = pov_player)
    _gui_generic_buttons.refresh(env)

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
        ppo_model = load_model(env, f'{trained_or_base(env.name)}.zip', device)
        agent_obj = Agent(f"Suggest Agent", ppo_model)
        agents.append(agent_obj)
    
    return agents

def trained_or_base(env_name, name='best_model'):
    """`name` if a trained model exists for `env_name` (zoo or pretrained), else 'base'."""
    for d in (os.path.join(config.MODELDIR, env_name), os.path.join(config.MODELDIR, 'pretrained', env_name)):
        if os.path.exists(os.path.join(d, f'{name}.zip')):
            return name
    return 'base'

@dataclass
class PlayOptions:
    suggest = False
    jamaica_players = 4

def create_game_page(env_class, env_name, agents_names, agent_load_names):
    # random seating order: shuffle display names and models together
    seats = list(zip(agents_names, agent_load_names))
    random.shuffle(seats)
    agents_names, agent_load_names = [list(s) for s in zip(*seats)]
    env = env_class(player_names=agents_names)
    # set seed
    seed = random.randint(0,1000)
    env.reset(seed = seed)
    # load agents
    agents = load_agents(env, agent_load_names, "cpu")
    # start gui
    env.nicegui_page()
    _gui_generic_buttons(env,)
    # play game
    play_step(env, agents, pov_player=agents_names.index('human'), suggest=app.storage.user["options"].suggest)

@ui.page('/frouge')
def frouge_page():
    from environments.frouge.envs.frouge import FlammeRougeEnv

    agents_names = ['human', 'best_model1', 'best_model2', 'best_model3', 'best_model4']
    create_game_page(FlammeRougeEnv, 'frouge', agents_names, ['human', 'best_model', 'best_model', 'best_model', 'best_model'])

@ui.page('/stotten')
def stotten_page():
    from environments.stotten.envs.stotten import SchottenTottenEnv

    agents_names = ['human', 'computer']
    create_game_page(SchottenTottenEnv, 'stotten', agents_names, ['human', 'best_model'])


@ui.page('/smallw')
def smallw_page():
    from environments.smallw.envs.smallw import SmallWorldEnv

    agents_names = ['human', 'computer 1', 'computer 2']
    create_game_page(SmallWorldEnv, 'smallw', agents_names, ['human', 'best_model', 'best_model'])


@ui.page('/jamaica')
def jamaica_page():
    from environments.jamaica.envs.jamaica import JamaicaEnv

    n_players = app.storage.user["options"].jamaica_players
    model = trained_or_base('jamaica')
    agents_names = ['human'] + [f'computer {i}' for i in range(1, n_players)]
    create_game_page(JamaicaEnv, 'jamaica', agents_names, ['human'] + [model] * (n_players - 1))


@ui.page('/')
def index():
    #init options on user scope
    app.storage.user["options"] = PlayOptions()

    ui.link('Flamme Rouge', frouge_page)
    ui.link('Schotten Totten', stotten_page)
    ui.link('Small World', smallw_page)
    with ui.row().classes('items-center'):
        ui.link('Jamaica', jamaica_page)
        ui.select({n: f'{n} players' for n in range(3, 7)}).bind_value(app.storage.user["options"], 'jamaica_players')
    with ui.row():
        ui.label('Suggest action:')
        ui.toggle({True:"Yes",False:"No"}).bind_value(app.storage.user["options"], 'suggest')

if __name__ in {"__main__", "__mp_main__"}:

    ui.run(title='GrosBill', storage_secret='private key almost impossible to guess')