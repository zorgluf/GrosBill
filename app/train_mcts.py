import os

import argparse
import time
import logging
import random
import gymnasium as gym
import numpy as np

from sb3_contrib import MaskablePPO as PPO1
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from utils.callbacks import SelfPlayCallback
from utils.files import reset_logs, reset_models, load_model
from utils.register import get_environment
from utils.selfplay import selfplay_wrapper

import config

from gymnasium import Env
from gymcts.gymcts_agent import GymctsAgent
from gymcts.gymcts_deepcopy_wrapper import DeepCopyMCTSGymEnvWrapper
from gymcts.logger import log

#TODO : insert a function that wraps the SelfPlayEnv so that action_space appear as a discrete one instead of a multi-descrete. Adapt env function accourdingly, for example for the step function.


log.setLevel(20)

env_name = "stotten"
model_dir = os.path.join(config.MODELDIR, env_name)

logger = logging.getLogger(__name__)
logger.setLevel(config.INFO)

logger.info('Setting up the selfplay training environment opponents...')
base_env = get_environment(env_name)

env = selfplay_wrapper(base_env)(opponent_type = "best", logger = logger, device = "cpu")
env.reset()

# 2. then wrap with the deep copy wrapper or a custom gymcts wrapper
env = DeepCopyMCTSGymEnvWrapper(env)
env.reset()

# 2. create the agent
agent = GymctsAgent(
    env=env,
    clear_mcts_tree_after_step=False,
    render_tree_after_step=True,
    number_of_simulations_per_step=50,
    exclude_unvisited_nodes_from_render=True
)

# Test the action space conversion
print(f"Original action space: {env.env.env.action_space}")
print(f"Wrapped action space: {env.env.action_space}")
print(f"Discrete action space size: {env.env.action_space.n}")

# 3. solve the environment
actions = agent.solve()

