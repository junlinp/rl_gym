import os
import json
import numpy as np

def save_episode_data(episode_data, env_name):
    # Create data directory if it doesn't exist
    if not os.path.exists(f'episode_data/{env_name}'):
        os.makedirs(f'episode_data/{env_name}')
    
    with open(f'episode_data/{env_name}/episode_data.json', 'w') as f:
        json.dump(episode_data, f)

def load_episode_data(path_dir:str):
    # find the meta.json file in the path_dir  
    # read the episode_num from the meta file
    with open(f'{path_dir}/episode_data.json', 'r') as f:
        data = json.load(f)
    return data