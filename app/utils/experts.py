import pickle
import os
from utils.register import get_trajectory_path

def save_trajectories(trajectories, env_name):
    path = get_trajectory_path(env_name)
    with open(os.path.join(path, 'trajectories.pkl'), 'wb') as f:
        pickle.dump(trajectories, f)

def load_trajectories(env_name):
    path = get_trajectory_path(env_name)
    trajectories = []
    if os.path.exists(os.path.join(path, 'trajectories.pkl')):
        with open(os.path.join(path, 'trajectories.pkl'), 'rb') as f:
            trajectories = pickle.load(f)
    return trajectories