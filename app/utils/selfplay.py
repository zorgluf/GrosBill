import os
import numpy as np
import random

from utils.files import load_model, load_all_models, get_best_model_name
from utils.agents import Agent
from utils.env import GBEnv


def selfplay_wrapper(env: GBEnv):
    class SelfPlayEnv(env):
        # wrapper over the normal single player env, but loads the best self play model
        def __init__(self, opponent_type, logger, device):
            super(SelfPlayEnv, self).__init__()
            self.device = device
            self.opponent_type = opponent_type
            self.opponent_models = load_all_models(self, device)
            self.best_model_name = get_best_model_name(self.name)
            self.logger = logger

        def setup_opponents(self):
            self.logger.debug(f'Setting up self play opponents for {self.name} with opponent type: {self.opponent_type}')
            # incremental load of new model
            best_model_name = get_best_model_name(self.name)
            if self.best_model_name != best_model_name:
                self.opponent_models.append(load_model(self, best_model_name, self.device ))
                self.best_model_name = best_model_name

            if self.opponent_type == 'random':
                start = 0
                end = len(self.opponent_models) - 1
                i = random.randint(start, end)
                self.opponent_agent = Agent('ppo_opponent', self.opponent_models[i]) 

            elif self.opponent_type == 'best':
                self.opponent_agent = Agent('ppo_opponent', self.opponent_models[-1])  

            elif self.opponent_type == 'mostly_best':
                j = random.uniform(0,1)
                if j < 0.7:
                    self.opponent_agent = Agent('ppo_opponent', self.opponent_models[-1])  
                else:
                    start = 0
                    end = len(self.opponent_models) - 1
                    i = random.randint(start, end)
                    self.opponent_agent = Agent('ppo_opponent', self.opponent_models[i])  

            elif self.opponent_type == 'base':
                self.opponent_agent = Agent('base', self.opponent_models[0])  

            self.agent_player_num = np.random.choice(self.n_players)
            self.agents = [self.opponent_agent] * self.n_players
            self.agents[self.agent_player_num] = None

        def reset(self, seed = None):
            _, info = super(SelfPlayEnv, self).reset(seed = seed)
            self.setup_opponents()

            if self.current_player != self.agent_player_num:   
                self.continue_game()

            return self.observation, info

        @property
        def current_agent(self):
            return self.agents[self.current_player]

        def continue_game(self):
            observation = None
            sum_reward = 0
            done = None
            truncated = False
            info = None

            while self.current_player != self.agent_player_num:
                action = self.current_agent.choose_action(self, choose_best_action = False, mask_invalid_actions = False)
                while True:
                    observation, reward, done, truncated, info = super(SelfPlayEnv, self).step(action)
                    sum_reward += reward[self.agent_player_num]
                    #continue if next step need no action
                    if (info['next_step_no_action'] == False) or done:
                        break
                    else:
                        action = -1  # no action, just continue the game
                if done:
                    break

            return observation, sum_reward, done, truncated, info


        def step(self, action):
            agent_reward = 0
            while True:
                observation, reward, done, truncated, info = super(SelfPlayEnv, self).step(action)
                agent_reward += reward[self.agent_player_num]
                #continue if next step need no action
                if (info['next_step_no_action'] == False) or done:
                    break
                else:
                    action = -1  # no action, just continue the game

            if not done:
                package = self.continue_game()
                if package[0] is not None:
                    observation, reward, done, truncated, info = package
                    agent_reward += reward

            return observation, agent_reward, done, truncated, info

    return SelfPlayEnv