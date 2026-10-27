import pytest

pytestmark = pytest.mark.nicegui_main_file('play.py')

from nicegui import app


@pytest.mark.parametrize('route', ['/', '/stotten'])
async def test_pages(user, route):
    await user.open(route)
    await user.should_see('GrosBill' if route == '/' else 'New game')

async def test_game_saved_and_resumed(user):
    await user.open('/stotten')
    await user.should_see('New game')
    saved = app.storage.user.get('saved_games', {}).get('stotten')
    assert saved is not None, 'game state must be saved on page open'
    state = saved['state']
    # reopening resumes the same game (same seat order, same state)
    await user.open('/stotten')
    await user.should_see('New game')
    saved2 = app.storage.user.get('saved_games', {}).get('stotten')
    assert saved2['state'] == state, 'resumed game must keep the same state'

async def test_new_game_clears_save(user):
    await user.open('/stotten')
    await user.should_see('New game')
    saved = app.storage.user.get('saved_games', {}).get('stotten')
    user.find('New game').click()
    await user.open('/stotten')
    await user.should_see('New game')
    saved2 = app.storage.user.get('saved_games', {}).get('stotten')
    assert saved2 is not None and saved2['state'] != saved['state'], 'a fresh game must be started'
