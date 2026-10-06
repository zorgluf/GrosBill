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
from pathlib import Path

app.add_static_files('/static', Path(__file__).parent / 'static')

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
            env.render(callback=lambda a: play_step(env, agents, pov_player, a, suggest=suggest), pov_player = pov_player)
            _gui_generic_buttons.refresh(env, callback=lambda a: play_step(env, agents, pov_player, a, suggest=suggest))
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
    smallw_players = 3
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
    from environments.smallw.envs.smallw import SmallWorldEnv, env_name_for

    # one board and one network per player count: zoo/smallw (3), zoo/smallw2/4/5
    n_players = app.storage.user["options"].smallw_players
    model = trained_or_base(env_name_for(n_players))
    agents_names = ['human'] + [f'computer {i}' for i in range(1, n_players)]
    create_game_page(lambda player_names: SmallWorldEnv(n_players, player_names),
                     env_name_for(n_players), agents_names, ['human'] + [model] * (n_players - 1))


@ui.page('/jamaica')
def jamaica_page():
    from environments.jamaica.envs.jamaica import JamaicaEnv

    n_players = app.storage.user["options"].jamaica_players
    model = trained_or_base('jamaica')
    agents_names = ['human'] + [f'computer {i}' for i in range(1, n_players)]
    create_game_page(JamaicaEnv, 'jamaica', agents_names, ['human'] + [model] * (n_players - 1))


GAMES = [
    # (page, env name, title, players, blurb)
    (frouge_page, 'frouge', 'Flamme Rouge', '5 players', 'Cycling race: pick energy cards for your rouleur and sprinter, draft behind the pack, beware of exhaustion.'),
    (stotten_page, 'stotten', 'Schotten Totten', '2 players', 'Claim the stones of the border by laying the best three-card formations on your side.'),
    (smallw_page, 'smallw', 'Small World', '2-5 players', 'Pick a race and power combo, conquer regions, go in decline and pick a fresh race.'),
    (jamaica_page, 'jamaica', 'Jamaica', '3-6 players', 'Pirate race around the island: load food, gold and powder, fight for the treasures.'),
]

def _smallw_env_name(n_players):
    from environments.smallw.envs.smallw import env_name_for
    return env_name_for(n_players)

#: Cards with a player-count selector: (counts, `PlayOptions` attribute, env name
#: (zoo directory) of a count). Small World trains one network per count,
#: Jamaica a single one for every count.
PLAYER_CHOICES = {
    'smallw': (range(2, 6), 'smallw_players', _smallw_env_name),
    'jamaica': (range(3, 7), 'jamaica_players', lambda n_players: 'jamaica'),
}

def _ai_badge(env_name):
    """'Trained AI' if a best_model.zip exists for `env_name` (zoo or pretrained)."""
    trained = trained_or_base(env_name) != 'base'
    badge = ui.badge('Trained AI' if trained else 'Untrained AI',
                     color='green-1' if trained else 'orange-1',
                     text_color='green-9' if trained else 'orange-9').classes('px-2 py-1')
    with badge:
        ui.tooltip(f'{env_name}: best_model.zip' if trained
                   else f'{env_name}: no best_model.zip yet, the AIs play the untrained base model')

def _game_card(page, env_name, title, players, blurb):
    choice = PLAYER_CHOICES.get(env_name)
    options = app.storage.user["options"]
    with ui.card().tight().classes('w-full hover:shadow-xl transition-shadow'):
        with ui.link(target=page).classes('w-full no-underline text-inherit'):
            ui.image(f'/static/screenshots/{env_name}.webp').props('ratio=1.6').classes('w-full')
            with ui.column().classes('gap-1 px-4 pt-3'):
                ui.label(title).classes('text-xl font-semibold text-gray-900')
                ui.label(blurb).classes('text-sm text-gray-600')
        with ui.row().classes('w-full items-center gap-2 px-4 pb-4 pt-2'):
            ui.badge(players, color='blue-grey-1', text_color='blue-grey-9').classes('px-2 py-1')
            ai_badge = ui.element('div')

            def show_ai_badge(n_players=None):
                # the trained state follows the selected player count
                ai_badge.clear()
                with ai_badge:
                    _ai_badge(env_name if choice is None else choice[2](n_players))

            show_ai_badge(None if choice is None else getattr(options, choice[1]))
            ui.space()
            if choice is not None:
                counts, attribute, _ = choice
                ui.select({n: f'{n} players' for n in counts},
                          on_change=lambda e: show_ai_badge(e.value)).props('dense outlined').bind_value(options, attribute)
            ui.button('Play', icon='play_arrow', on_click=lambda: ui.navigate.to(page)).props('unelevated')

@ui.page('/')
def index():
    #init options on user scope
    app.storage.user["options"] = PlayOptions()

    ui.query('body').classes('bg-slate-100')
    with ui.column().classes('w-full max-w-6xl mx-auto p-6 gap-6'):
        with ui.row().classes('w-full items-center gap-4'):
            ui.image('/static/logo.png').classes('w-16 h-16')
            with ui.column().classes('gap-0'):
                ui.label('GrosBill').classes('text-3xl font-bold text-gray-900')
                ui.label('Play board games against self-play trained agents').classes('text-gray-600')
            ui.space()
            ui.switch('Suggest actions').bind_value(app.storage.user["options"], 'suggest')
        with ui.grid().classes('w-full grid-cols-1 md:grid-cols-2 gap-6'):
            for game in GAMES:
                _game_card(*game)

if __name__ in {"__main__", "__mp_main__"}:

    ui.run(title='GrosBill', storage_secret='private key almost impossible to guess')