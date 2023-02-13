import d3rlpy
import json
import numpy as np
import pandas as pd
import random
from zipfile import ZipFile
from gym import spaces, Env
import yaml
import pickle
import shutil
from datetime import datetime
import re
import csv
import codecs
import os
import sys
import torch
from gym_env_demo import get_env
from mpc_policy import get_mpc_controller
import matplotlib.pyplot as plt
from tqdm import tqdm



def report_rewards(env, rewards_list, algo_names=None, save_dir=None):
    result_dict = {}
    if algo_names is None:
        algo_names = []
        for i in range(len(rewards_list)):
            algo_names.append(f'algo_{i}')
    num_episodes = len(rewards_list[0])
    for n_algo in range(len(algo_names)):
        algo_name = algo_names[n_algo]
        rewards_list_curr_algo = rewards_list[n_algo]
        rewards_mean_over_episodes = []  # rewards_mean_over_episodes[n_epi] is mean of rewards of n_epi
        for n_epi in range(num_episodes):
            if rewards_list_curr_algo[n_epi][-1] == env.error_reward:
                rewards_mean_over_episodes.append(env.error_reward)
            else:
                rewards_mean_over_episodes.append(np.mean(rewards_list_curr_algo[n_epi]))
        on_episodes_reward_mean = np.mean(rewards_mean_over_episodes)
        on_episodes_reward_std = np.std(rewards_mean_over_episodes)
        unwrap_list = []
        for games_r_list in rewards_list_curr_algo:
            unwrap_list += list(games_r_list)
        all_reward_mean = np.mean(unwrap_list)
        all_reward_std = np.std(unwrap_list)
        print(f"{algo_name}_on_episodes_reward_mean: {on_episodes_reward_mean}")
        result_dict[algo_name + "_on_episodes_reward_mean"] = on_episodes_reward_mean
        print(f"{algo_name}_on_episodes_reward_std: {on_episodes_reward_std}")
        result_dict[algo_name + "_on_episodes_reward_std"] = on_episodes_reward_std
        print(f"{algo_name}_all_reward_mean: {all_reward_mean}")
        result_dict[algo_name + "_all_reward_mean"] = all_reward_mean
        print(f"{algo_name}_all_reward_std: {all_reward_std}")
        result_dict[algo_name + "_all_reward_std"] = all_reward_std
    os.makedirs(save_dir, exist_ok=True)
    f_dir = os.path.join(save_dir, 'result.json')
    json.dump(result_dict, open(f_dir, 'w+'))
    return result_dict

# ---- standard ----
def evalute_algorithms(env, algorithms, num_episodes=1, to_plt=True,
                        plot_dir='./plt_results'):
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
    initial_states = []
    seeds = list(range(num_episodes))
    for seed in seeds: initial_states.append(env.reset(seed=seed))
    
    observations_list = [[] for _ in range(len(algorithms))] 
    actions_list = [[] for _ in range(len(algorithms))]
    rewards_list = [[] for _ in range(len(algorithms))]
    for n_epi in tqdm(range(num_episodes)):
        for n_algo in range(len(algorithms)):
            algo, algo_name = algorithms[n_algo]
            algo_observes = []
            algo_actions = []
            algo_rewards = []  # list, for this algorithm, reawards of this trajectory.
            init_obs = env.reset(init_state=initial_states[n_epi])
            o = [init_obs]
            done = False
            while not done:
                a = algo.predict(o)
                algo_actions.append(a[0])
                o, r, done, _ = env.step(a)
                algo_observes.append(o)
                algo_rewards.append(r)
                o = [o]
            observations_list[n_algo].append(algo_observes)
            actions_list[n_algo].append(algo_actions)
            rewards_list[n_algo].append(algo_rewards)

        if to_plt:
            # plot observations
            for n_o in range(env.observation_dim):
                o_name = env.observation_name[n_o]

                plt.close("all")
                plt.figure(0)
                plt.title(f"{o_name}")
                for n_algo in range(len(algorithms)):
                    alpha = 1 * (0.7 ** (len(algorithms) - 1 - n_algo))
                    _, algo_name = algorithms[n_algo]
                    plt.plot(np.array(observations_list[n_algo][-1])[:, n_o], label=algo_name, alpha=alpha)
                plt.plot([initial_states[n_epi][n_o] for _ in range(env.episode_len)], linestyle="--",
                            label=f"initial_{o_name}")
                plt.plot([env.steady_observation[n_o] for _ in range(env.episode_len)], linestyle="-.",
                            label=f"steady_{o_name}")
                plt.xticks(np.arange(1, env.episode_len + 2, 1))
                plt.annotate(str(initial_states[n_epi][n_o]), xy=(0, initial_states[n_epi][n_o]))
                plt.annotate(str(env.steady_observation[n_o]), xy=(0, env.steady_observation[n_o]))
                plt.legend()
                if plot_dir is not None:
                    path_name = os.path.join(plot_dir, f"{n_epi}_observation_{o_name}.png")
                    plt.savefig(path_name)
                plt.close()

            # plot actions
            for n_a in range(env.action_dim):
                a_name = env.action_name[n_a]
                plt.close("all")
                plt.figure(0)
                plt.title(f"{a_name}")
                for n_algo in range(len(algorithms)):
                    alpha = 1 * (0.7 ** (len(algorithms) - 1 - n_algo))
                    _, algo_name = algorithms[n_algo]
                    plt.plot(np.array(actions_list[n_algo][-1])[:, n_a], label=algo_name, alpha=alpha)
                plt.plot([env.steady_action[n_a] for _ in range(env.episode_len)], linestyle="-.",
                            label=f"steady_{a_name}")
                plt.xticks(np.arange(1, env.episode_len + 2, 1))
                plt.legend()
                if plot_dir is not None:
                    path_name = os.path.join(plot_dir, f"{n_epi}_action_{a_name}.png")
                    plt.savefig(path_name)
                plt.close()

            # plot rewards
            plt.close("all")
            plt.figure(0)
            plt.title("reward")
            for n_algo in range(len(algorithms)):
                alpha = 1 * (0.7 ** (len(algorithms) - 1 - n_algo))
                _, algo_name = algorithms[n_algo]
                plt.plot(np.array(rewards_list[n_algo][-1]), label=algo_name, alpha=alpha)
            plt.xticks(np.arange(1, env.episode_len + 2, 1))
            plt.legend()
            if plot_dir is not None:
                path_name = os.path.join(plot_dir, f"{n_epi}_reward.png")
                plt.savefig(path_name)
            plt.close()

    observations_list = np.array(observations_list)
    actions_list = np.array(actions_list)
    rewards_list = np.array(rewards_list)
    return observations_list, actions_list, rewards_list


class OfflineRLModel(object):
    def __init__(self, algo_name, dir_loc, config_dict_loc='offline_experiments.yaml'):
        with open(os.path.join(dir_loc, config_dict_loc), 'r') as fp:
            config_dict = yaml.safe_load(fp)
        best_loc = os.path.join(dir_loc, config_dict['best_loc'])
        best_params = os.path.join(best_loc, f'{algo_name}/params.json')
        best_ckpt = os.path.join(best_loc, f'{algo_name}/best.pt')

        if algo_name == 'CQL':
            curr_algo = d3rlpy.algos.CQL.from_json(best_params)
        elif algo_name == 'PLAS':
            curr_algo = d3rlpy.algos.PLAS.from_json(best_params)
        elif algo_name == 'PLASWithPerturbation':
            curr_algo = d3rlpy.algos.PLASWithPerturbation.from_json(best_params)
        elif algo_name == 'DDPG':
            curr_algo = d3rlpy.algos.DDPG.from_json(best_params)
        elif algo_name == 'BC':
            curr_algo = d3rlpy.algos.BC.from_json(best_params)
        elif algo_name == 'TD3':
            curr_algo = d3rlpy.algos.TD3.from_json(best_params)
        elif algo_name == 'BEAR':
            curr_algo = d3rlpy.algos.BEAR.from_json(best_params)
        elif algo_name == 'SAC':
            curr_algo = d3rlpy.algos.SAC.from_json(best_params)
        elif algo_name == 'BCQ':
            curr_algo = d3rlpy.algos.BCQ.from_json(best_params)
        elif algo_name == 'CRR':
            curr_algo = d3rlpy.algos.CRR.from_json(best_params)
        elif algo_name == 'AWR':
            curr_algo = d3rlpy.algos.AWR.from_json(best_params)
        elif algo_name == 'AWAC':
            curr_algo = d3rlpy.algos.AWAC.from_json(best_params)
        elif algo_name == 'COMBO':
            curr_algo = d3rlpy.algos.COMBO.from_json(best_params)
        elif algo_name == 'MOPO':
            curr_algo = d3rlpy.algos.MOPO.from_json(best_params)
        elif algo_name == 'TD3PlusBC':
            curr_algo = d3rlpy.algos.TD3PlusBC.from_json(best_params)
        else:
            raise Exception("algo_name is invalid!", algo_name)

        curr_algo.load_model(best_ckpt)
        self.curr_algo = curr_algo

    def predict(self, state):
        state = torch.from_numpy(np.asarray(state))
        if self.curr_algo.scaler is not None: state = self.curr_algo.scaler.transform(state)
        inp = self.curr_algo.predict(state)  # shape (1,x)
        if self.curr_algo.action_scaler is not None:
            inp = torch.from_numpy(inp)
            inp = self.curr_algo.action_scaler.reverse_transform(inp).cpu().numpy()
        return inp


dir_loc = os.path.dirname(os.path.relpath(__file__))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--plt', action='store_true', help="whether plot")
parser.add_argument('--algo', type=str, help='algorithm', default="CQL")
parser.add_argument('--exp', type=str, help='exp. name', default="random")
parser.add_argument("--device", type=int, help='device id', default="0")
parser.add_argument("--num_episodes", type=int, help='num. of episodes', default="1")

if __name__ == "__main__":
    args = parser.parse_args()
    data_loc = os.environ['AMLT_DATA_DIR'] if args.amlt else dir_loc
    dataset_dir = os.path.join(data_loc, 'datasets')
    use_gpu = (False if args.device < 0 else args.device)
    with open(os.path.join(dir_loc, 'offline_experiments.yaml'), 'r') as fp:
        config_dict = yaml.safe_load(fp)
    seed = config_dict['seed']
    num_of_seeds = config_dict['num_of_seeds']
    env_name = config_dict['env_name']
    model_name = config_dict['model_name']
    dense_reward = config_dict['dense_reward']
    debug_mode = config_dict['debug_mode']

    # for offlineRL online learning
    online_training = config_dict['online_training']
    buffer_maxlen = config_dict['buffer_maxlen']
    explorer_start_epsilon = config_dict['explorer_start_epsilon']
    explorer_end_epsilon = config_dict['explorer_end_epsilon']
    explorer_duration = config_dict['explorer_duration']
    n_steps_per_epoch = config_dict['n_steps_per_epoch']
    online_random_steps = config_dict['online_random_steps']
    online_update_interval = config_dict['online_update_interval']
    online_save_interval = config_dict['online_save_interval']

    # for offline data generation and training
    N_EPOCHS  = config_dict['N_EPOCHS']
    DYNAMICS_N_EPOCHS = config_dict['DYNAMICS_N_EPOCHS']
    scaler = config_dict['scaler']
    action_scaler = config_dict['action_scaler']
    reward_scaler = config_dict['reward_scaler']
    evaluate_on_environment = config_dict['evaluate_on_environment']
    default_loc = os.path.join(data_loc, config_dict['default_loc'])
    plt_dir = os.path.join(data_loc, config_dict['plt_dir'])
    dataset_location = os.path.join(data_loc, config_dict['dataset_location'])
    training_dataset_loc = os.path.join(data_loc, config_dict['training_dataset_loc'])
    eval_dataset_loc = os.path.join(data_loc, config_dict['eval_dataset_loc'])
    test_initial_states = os.path.join(data_loc, config_dict['test_initial_states'])

    # env specific configs
    reward_on_steady = config_dict.get('reward_on_steady', None)
    reward_on_absolute_efactor = config_dict.get('reward_on_absolute_efactor', None)
    compute_diffs_on_reward = config_dict.get('compute_diffs_on_reward', None)
    standard_reward_style = config_dict.get('standard_reward_style', None)
    initial_state_deviation_ratio = config_dict.get('initial_state_deviation_ratio', None)

    if seed is not None:
        seeds = [seed]
    else:
        num_of_seeds = config_dict['num_of_seeds']
        seeds = []
        for i in range(num_of_seeds):
            seeds.append(random.randint(0, 2**32-1))
    
    env = get_env()
    # algo_names = ['PID', "MPC", 'BC', 'CQL', 'PLAS', 'PLASWithPerturbation', 'BEAR', 'SAC', 'BCQ', 'CRR', 'AWR', 'AWAC', 'DDPG', 'TD3', 'COMBO', 'MOPO']
    algo_names = args.algo.split(",")
    results_csv = ['algo_name', 'on_episodes_reward_mean', 'episodes_reward_std', 'all_reward_mean', 'all_reward_std']
    try:
        for algo_name in algo_names:
            num_episodes = args.num_episodes
            if algo_name == 'MPC':
                baseline_model = get_mpc_controller(env)
                baseline_model_name = "MPC"
                algorithms = [(baseline_model, baseline_model_name)]
            else:
                curr_algo = OfflineRLModel(algo_name, dir_loc, config_dict_loc=f'OFFLINE_BEST.yaml')
                algorithms = [(curr_algo, algo_name)]
            save_dir = os.path.join(plt_dir, algo_name)
            observations_list, actions_list, rewards_list = evalute_algorithms(env, algorithms, num_episodes=num_episodes, to_plt=args.plt, plot_dir=save_dir)
            results_dict = report_rewards(env, rewards_list, algo_names=[_a_name for _, _a_name in algorithms], save_dir=save_dir)
            results_csv.append([algo_name, results_dict[f'{algo_name}_on_episodes_reward_mean'], results_dict[f'{algo_name}_on_episodes_reward_std'], results_dict[f'{algo_name}_all_reward_mean'], results_dict[f'{algo_name}_all_reward_std']])
            np.save(os.path.join(save_dir, f'observations.npy'), observations_list)
            np.save(os.path.join(save_dir, f'actions.npy'), actions_list)
            np.save(os.path.join(save_dir, f'rewards.npy'), rewards_list)
    except FileNotFoundError as e:
        print(e)
    with codecs.open(os.path.join(plt_dir, "total_results_dict.csv"), "w+", encoding="utf-8") as fp:
        csv_writer = csv.writer(fp)
        for row in results_csv:
            csv_writer.writerow(row)

