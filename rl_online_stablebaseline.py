import argparse
import importlib
import json
import os
import random
import sys
from datetime import datetime
from zipfile import ZipFile

import numpy as np
import pandas as pd
import stable_baselines3 as SBS
import yaml


def evaluate(model, eval_env, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = eval_env.reset(seed=i+7)
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = eval_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
    return mean_episode_reward



dir_loc = os.path.dirname(os.path.relpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--algo', type=str, help='algorithm', default="CQL")
parser.add_argument('--exp', type=str, help='exp. name', default="random")
parser.add_argument("--device", type=int, help='device id', default="0")
parser.add_argument("--iter", type=int, help='iter. num.', default=0)
parser.add_argument("--env", type=str, help='env. name', default="CSTR")


# TODO: load previous algorithm before training
if __name__ == "__main__":
    args = parser.parse_args()
    with open(os.path.join(dir_loc, 'experiments.yaml'), 'r') as fp:
        config_dict = yaml.safe_load(fp)
    seed = config_dict['seed']
    num_of_seeds = config_dict['num_of_seeds']
    env_name = args.env # config_dict['env_name']
    model_name = config_dict['model_name']
    dense_reward = config_dict['dense_reward']
    debug_mode = config_dict['debug_mode']
    
    data_loc = os.environ['AMLT_DATA_DIR'] if args.amlt else dir_loc
    dataset_dir = os.path.join(data_loc, 'datasets', env_name)
    use_gpu = (False if args.device < 0 else args.device)

    env_lib = importlib.import_module(f"examples.{args.env}.template_env")
    env = env_lib.get_env()
    eval_env = env_lib.get_env()
    
    total_learning_steps = 10000000
    eval_steps = 100000
    model = SBS.PPO('MlpPolicy', env, verbose=1, device=("cpu" if not use_gpu else f'cuda:{args.device}'), n_steps=32)
    reward_list = []
    best_reward = float("-inf")
    for i in range(total_learning_steps//eval_steps):
        model.learn(total_timesteps=eval_steps, reset_num_timesteps=True)
        mean_reward_before_train = evaluate(model, eval_env, num_episodes=10)
        print("eval: ", i, mean_reward_before_train)
        reward_list.append(mean_reward_before_train)
        if np.max(reward_list) > best_reward:
            best_reward = np.max(reward_list)
            model.save("PPO_online_best.pt")