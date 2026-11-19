import copy
import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import random
import string
import logging as logger

from sb3_contrib.common.maskable.utils import get_action_masks

def sample_action(action_probs):
    action = np.random.choice(len(action_probs), p = action_probs)
    return action


def mask_actions(legal_actions, action_probs):
    masked_action_probs = np.multiply(legal_actions, action_probs)
    masked_action_probs = masked_action_probs / np.sum(masked_action_probs)
    return masked_action_probs


class Agent():
  def __init__(self, name, model = None):
      self.name = name
      self.id = self.name + '_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
      self.model = model
      self.points = 0
      if model != None:
        self.device = model.device

  def print_top_actions(self, action_probs):
    top5_action_idx = np.argsort(-action_probs)[:5]
    top5_actions = action_probs[top5_action_idx]
    logger.debug(f"Top 5 actions: {[str(i) + ': ' + str(round(a,2))[:5] for i,a in zip(top5_action_idx, top5_actions)]}")

  def choose_action(self, env, choose_best_action):
    action_masks = get_action_masks(env)
    action = self.model.predict(env.observation, deterministic = choose_best_action, action_masks = action_masks)[0]
    logger.debug(f'Best action {action}')

    return action


class RandomAgent(Agent):
  """Plays a uniformly random legal action: the stand-in for the games that have no
  policy yet (agent name `random` in test.py / play.py)."""

  def __init__(self, name = 'random'):
      super().__init__(name)

  def choose_action(self, env, choose_best_action):
    return int(np.random.choice(np.flatnonzero(get_action_masks(env))))


class GreedyAgent(Agent):
  """One-ply greedy on the env's own reward: plays a copy of the env for every legal
  action and keeps the one that pays the mover most (ties at random). A scripted
  reference that does not drift like the self-play generations (agent name `greedy`
  in test.py / play.py). It steps the real state, so a random effect or a hidden tile
  that ends the game is visible to it: a yardstick, not a fair player."""

  def __init__(self, name = 'greedy'):
      super().__init__(name)

  @staticmethod
  def scores(env):
    """{legal action: immediate reward of the player to move}."""
    player = env.current_player
    step = getattr(env, 'apply', None)
    out = {}
    for action in np.flatnonzero(get_action_masks(env)):
      trial = copy.deepcopy(env)
      reward = trial.apply(int(action)) if step is not None else trial.step(int(action))[1]
      out[int(action)] = float(reward[player])
    return out

  def choose_action(self, env, choose_best_action):
    scores = self.scores(env)
    best = max(scores.values())
    return int(np.random.choice([a for a, s in scores.items() if s >= best - 1e-9]))
