import asyncio

import pytest

pytestmark = pytest.mark.nicegui_main_file('play.py')

from nicegui import app, ui


@pytest.mark.parametrize('route', ['/', '/stotten'])
async def test_pages(user, route):
    await user.open(route)
    await user.should_see('GrosBill' if route == '/' else 'hand')

async def test_game_saved_and_resumed(user):
    await user.open('/stotten')
    await user.should_see('hand')
    saved = app.storage.user.get('saved_games', {}).get('stotten')
    assert saved is not None, 'game state must be saved on page open'
    state = saved['state']
    # reopening resumes the same game (same seat order, same state)
    await user.open('/stotten')
    await user.should_see('hand')
    saved2 = app.storage.user.get('saved_games', {}).get('stotten')
    assert saved2['state'] == state, 'resumed game must keep the same state'

async def _open_drop_dialog(user):
    """Click the cross of the 'Game in progress' badge and wait for the dialog."""
    user.find(kind=ui.button, content='close').click()
    for _ in range(40):
        if any(d.value for d in user.find(kind=ui.dialog).elements):
            return
        await asyncio.sleep(0.05)
    raise AssertionError('the confirmation dialog did not open')

async def test_abandon_saved_game(user):
    await user.open('/stotten')
    await user.should_see('hand')
    await user.open('/')
    await user.should_see('Game in progress')
    # cancelling the confirmation keeps the game
    await _open_drop_dialog(user)
    await user.should_see('Abandon your Schotten Totten game in progress?')
    user.find('Cancel').click()
    await user.should_see('Game in progress')
    assert 'stotten' in app.storage.user['saved_games']
    # confirming drops it
    await _open_drop_dialog(user)
    await user.should_see('Abandon your Schotten Totten game in progress?')
    user.find('Abandon game').click()
    await user.should_not_see('Game in progress')
    assert 'stotten' not in app.storage.user['saved_games']

async def test_options_from_disk(user):
    # the user storage is reloaded from JSON after a server restart: the options
    # come back as a plain dict, possibly without every key (old format: {})
    await user.open('/')
    app.storage.user['options'] = {}
    await user.open('/')
    await user.should_see('GrosBill')
    await user.open('/smallw')
    await user.should_see('hand')

async def test_badge_follows_player_count(user):
    await user.open('/')
    app.storage.user['options']['smallw_players'] = 3
    await user.open('/smallw')
    await user.should_see('hand')
    assert 'smallw' in app.storage.user['saved_games']
    await user.open('/')
    await user.should_see('Game in progress')
    # no saved game for the 4-player board: no badge on the Small World card
    app.storage.user['options']['smallw_players'] = 4
    await user.open('/')
    await user.should_see('GrosBill')
    await user.should_not_see('Game in progress')
