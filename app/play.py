from nicegui import ui, app
import random
import os
import shutil
import config
from utils.agents import Agent
from utils.files import load_model
from typing import List, Tuple
from utils.env import GBEnv
from pathlib import Path

app.add_static_files('/static', Path(__file__).parent / 'static')

@ui.refreshable
def _gui_generic_buttons(env: GBEnv, callback = None):
    ui.button("Next step", on_click=lambda: callback(None)).bind_visibility_from(env, "current_player", backward=lambda current_player: current_player == -1)
    with ui.dialog() as dialog, ui.card():
        if env.done:
            ranking = env.final_ranking()
            if len(ranking) == 0:
                ui.label("Winner undetermined.")
            for rank, player_idx in enumerate(ranking):
                ui.label(f"{rank+1}. {env.player_names[player_idx]}")
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
        if done:
            _clear_saved_game(env.name)
        else:
            _save_game(env)
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

    if _options()['suggest']:
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

#: Key in the user storage holding the saved games, one entry per env name.
#: `app.storage.user` is kept on the server, keyed by the browser session: the
#: game survives a disconnection and a browser restart that restores the session.
SAVED_GAMES_KEY = 'saved_games'

def _save_game(env: GBEnv):
    """Persist the current game in the user storage so it survives a reload."""
    saved_games = app.storage.user.get(SAVED_GAMES_KEY, {})
    saved_games[env.name] = {
        'state': env.save_state(),
        'agents_names': env.player_names,
        'agent_load_names': env.play_meta['agent_load_names'],
        'pov_player': env.play_meta['pov_player'],
        'n_players': env.n_players,
    }
    app.storage.user[SAVED_GAMES_KEY] = saved_games

def _saved_game(env_name: str):
    return app.storage.user.get(SAVED_GAMES_KEY, {}).get(env_name)

def _clear_saved_game(env_name: str):
    saved_games = app.storage.user.get(SAVED_GAMES_KEY, {})
    if env_name in saved_games:
        del saved_games[env_name]
        app.storage.user[SAVED_GAMES_KEY] = saved_games

def create_game_page(make_env, env_name, agents_names, agent_load_names):
    """Set up a game page. `make_env(n_players, player_names)` builds the env.
    If a saved game exists for `env_name` in the browser storage, it is resumed."""
    saved = _saved_game(env_name)
    if saved is not None:
        try:
            agents_names = saved['agents_names']
            agent_load_names = saved['agent_load_names']
            env = make_env(saved['n_players'], agents_names)
            env.restore_state(saved['state'])
            pov_player = saved['pov_player']
        except Exception:
            # stale or incompatible save (e.g. env changed since): start fresh
            _clear_saved_game(env_name)
            saved = None
    if saved is None:
        # random seating order: shuffle display names and models together
        seats = list(zip(agents_names, agent_load_names))
        random.shuffle(seats)
        agents_names, agent_load_names = [list(s) for s in zip(*seats)]
        env = make_env(len(agents_names), agents_names)
        # set seed
        seed = random.randint(0,1000)
        env.reset(seed = seed)
        pov_player = agents_names.index('human')
    # load agents
    agents = load_agents(env, agent_load_names, "cpu")
    env.play_meta = {'pov_player': pov_player, 'agent_load_names': agent_load_names}
    # start gui
    env.nicegui_page()
    _gui_generic_buttons(env,)
    if saved is None:
        _save_game(env)
    # play game
    play_step(env, agents, pov_player=pov_player, suggest=_options()['suggest'])

#: Default per-user play options. They are kept as a plain dict: the user storage
#: is persisted as JSON, so an object would come back as a dict after a restart.
DEFAULT_OPTIONS = {'suggest': False, 'smallw_players': 3, 'jamaica_players': 4}

def _options() -> dict:
    """The per-user play options, missing keys filled with their default."""
    options = app.storage.user.setdefault('options', {})
    for key, value in DEFAULT_OPTIONS.items():
        options.setdefault(key, value)
    return options

@ui.page('/frouge')
def frouge_page():
    from environments.frouge.envs.frouge import FlammeRougeEnv

    agents_names = ['human', 'best_model1', 'best_model2', 'best_model3', 'best_model4']
    create_game_page(lambda n, names: FlammeRougeEnv(names), 'frouge', agents_names,
                     ['human', 'best_model', 'best_model', 'best_model', 'best_model'])

@ui.page('/stotten')
def stotten_page():
    from environments.stotten.envs.stotten import SchottenTottenEnv

    agents_names = ['human', 'computer']
    create_game_page(lambda n, names: SchottenTottenEnv(names), 'stotten', agents_names, ['human', 'best_model'])


@ui.page('/smallw')
def smallw_page():
    from environments.smallw.envs.smallw import SmallWorldEnv, env_name_for

    # one board and one network per player count: zoo/smallw (3), zoo/smallw2/4/5
    n_players = _options()['smallw_players']
    model = trained_or_base(env_name_for(n_players))
    agents_names = ['human'] + [f'computer {i}' for i in range(1, n_players)]
    create_game_page(lambda n, names: SmallWorldEnv(n, names),
                     env_name_for(n_players), agents_names, ['human'] + [model] * (n_players - 1))


@ui.page('/jamaica')
def jamaica_page():
    from environments.jamaica.envs.jamaica import JamaicaEnv

    n_players = _options()['jamaica_players']
    model = trained_or_base('jamaica')
    agents_names = ['human'] + [f'computer {i}' for i in range(1, n_players)]
    create_game_page(lambda n, names: JamaicaEnv(n, names), 'jamaica', agents_names,
                     ['human'] + [model] * (n_players - 1))


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

#: Cards with a player-count selector: (counts, options key, env name
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

def _confirm_drop_dialog():
    """The home page dialog confirming the deletion of a saved game: returns an
    async function `confirm(title)`, True if the player confirmed."""
    with ui.dialog() as dialog, ui.card():
        label = ui.label()
        with ui.row().classes('w-full justify-end'):
            ui.button('Cancel', on_click=dialog.close).props('flat')
            ui.button('Abandon game', color='negative', on_click=lambda: dialog.submit(True))

    async def confirm(title):
        label.text = f'Abandon your {title} game in progress?'
        return await dialog
    return confirm

def _game_card(page, env_name, title, players, blurb, confirm_drop):
    choice = PLAYER_CHOICES.get(env_name)
    options = _options()
    with ui.card().tight().classes('w-full hover:shadow-xl transition-shadow'):
        with ui.link(target=page).classes('w-full no-underline text-inherit'):
            ui.image(f'/static/screenshots/{env_name}.webp').props('ratio=1.6').classes('w-full')
            with ui.column().classes('gap-1 px-4 pt-3'):
                ui.label(title).classes('text-xl font-semibold text-gray-900')
                ui.label(blurb).classes('text-sm text-gray-600')
        with ui.row().classes('w-full items-center gap-2 px-4 pb-4 pt-2'):
            ui.badge(players, color='blue-grey-1', text_color='blue-grey-9').classes('px-2 py-1')
            badges = ui.row().classes('items-center gap-2')

            async def drop_saved_game(count_env_name, n_players):
                if await confirm_drop(title):
                    _clear_saved_game(count_env_name)
                    show_badges(n_players)

            def show_badges(n_players=None):
                # the trained state and the saved game follow the selected player count
                count_env_name = env_name if choice is None else choice[2](n_players)
                badges.clear()
                with badges:
                    _ai_badge(count_env_name)
                    if _saved_game(count_env_name) is not None:
                        with ui.badge('Game in progress', color='orange', text_color='white').classes('px-2 py-1 gap-1'):
                            ui.tooltip('Your saved game will resume when you click Play')
                            ui.button(icon='close', on_click=lambda: drop_saved_game(count_env_name, n_players)) \
                                .props('flat round dense size=xs color=white').tooltip('Abandon this game')

            show_badges(None if choice is None else options[choice[1]])
            ui.space()
            if choice is not None:
                counts, key, _ = choice
                ui.select({n: f'{n} players' for n in counts},
                          on_change=lambda e: show_badges(e.value)).props('dense outlined').bind_value(options, key)
            ui.button('Play', icon='play_arrow', on_click=lambda: ui.navigate.to(page)).props('unelevated')

@ui.page('/')
def index():
    #init options on user scope
    _options()

    ui.query('body').classes('bg-slate-100')
    with ui.column().classes('w-full max-w-6xl mx-auto p-6 gap-6'):
        with ui.row().classes('w-full items-center gap-4'):
            ui.image('/static/logo.png').classes('w-16 h-16')
            with ui.column().classes('gap-0'):
                ui.label('GrosBill').classes('text-3xl font-bold text-gray-900')
                ui.label('Play board games against self-play trained agents').classes('text-gray-600')
            ui.space()
            ui.switch('Suggest actions').bind_value(_options(), 'suggest')
        with ui.grid().classes('w-full grid-cols-1 md:grid-cols-2 gap-6'):
            confirm_drop = _confirm_drop_dialog()
            for game in GAMES:
                _game_card(*game, confirm_drop)

if __name__ in {"__main__", "__mp_main__"}:

    ui.run(title='GrosBill', favicon=Path(__file__).parent / 'static' / 'favicon.ico',
           storage_secret='private key almost impossible to guess')