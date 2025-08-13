from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .stotten import SchottenTottenEnv

from nicegui import ui

from .classes import *

def _action_play_card(stone_idx:int, play_callback, env: SchottenTottenEnv):
    if env.gui_hand_card_selected == None:
        ui.notify("Please select a card in your hand before choosing the stone")
    else:
        play_callback([ env.gui_hand_card_selected, stone_idx])

@ui.refreshable
def _get_card_element(card: Card):
    with ui.card().tight().style("width:10vw; aspect-ratio: 2 / 3;"):
        ui.label(f"{card.value}").style("font-size: 10vw; width:100%; text-align: center; line-height: normal").tailwind.text_color(card.color.name.lower())
        ui.label(f"{card.value}").style("width: 10%; position: absolute; top: 0px; right: 0px;").tailwind.text_color(card.color.name.lower())

@ui.refreshable
def _gui_board(env, callback):
    board = env.board
    with ui.grid(columns=NB_STONES).classes("gap-0"):
        # Opponent's cards on stone
        for i in range(NB_STONES):
            with ui.column().classes("gap-0").style("position: relative; height: 15vw;"):
                for i, c in enumerate(board.played_cards[i][abs(env.current_player -1)]):
                    with ui.element("div").style(f"position: absolute; margin-top: {i * 15}%;"):
                        _get_card_element(c)
        # Stone position
        for i in range(NB_STONES):
            with ui.column().on("click",lambda i=i: _action_play_card(i, callback, env)).classes("gap-2"):
                if (env.current_player == PlayerId.PLAYER1.value and board.stones[i] == StonePosition.PLAYER2) or (env.current_player == PlayerId.PLAYER2.value and board.stones[i] == StonePosition.PLAYER1):
                    ui.element("div").style("width: 10vw; height:10vh").tailwind.background_color("grey")
                else:
                    ui.element("div").style("width: 10vw; height:10vh; border: 2px dashed grey;")
                if board.stones[i] == StonePosition.NEUTRAL:
                    ui.element("div").style("width: 10vw; height:10vh").tailwind.background_color("grey")
                else:
                    ui.element("div").style("width: 10vw; height:10vh; border: 2px dashed grey;")
                if (env.current_player == PlayerId.PLAYER1.value and board.stones[i] == StonePosition.PLAYER1) or (env.current_player == PlayerId.PLAYER2.value and board.stones[i] == StonePosition.PLAYER2):
                    ui.element("div").style("width: 10vw; height:10vh").tailwind.background_color("grey")
                else:
                    ui.element("div").style("width: 10vw; height:10vh; border: 2px dashed grey;")
        # Current player's cards on stone
        for i in range(NB_STONES):
            with ui.column().classes("gap-0").style("position: relative; height: 15vw;"):
                for i, c in enumerate(board.played_cards[i][env.current_player]):
                    with ui.element("div").style(f"position: absolute; margin-top: {i * 15}%;"):
                        _get_card_element(c)

@ui.refreshable
def _gui_hand(env: SchottenTottenEnv, callback, suggested_action = None):
    ui.label().bind_text_from(env, 'current_player', backward=lambda i: f"Player {env.board.players[i].name} hand")
    with ui.row().bind_visibility_from(env, 'done', backward=lambda x: not x):
        cards = env.board.players[env.current_player].hand
        card_toggle = ui.toggle(dict(enumerate([''] * len(cards)))).bind_value_to(env,"gui_hand_card_selected")
        for i, card in enumerate(env.board.players[env.current_player].hand):
            with ui.teleport(f'#{card_toggle.html_id} > button:nth-child({i+1})'):
                _get_card_element(card)

def init_web(env: SchottenTottenEnv, callback = None):
    with ui.column():
        _gui_board(env, callback)
        _gui_hand(env, callback)


def render_web(env: SchottenTottenEnv, callback, suggested_action = None, **kwargs):
    _gui_board.refresh(env, callback)
    _gui_hand.refresh(env, callback, suggested_action)
    #TODO handle finish game (do a last update on ui and print the winner)


    
