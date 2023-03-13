import sys
import os
import torch
import d3rlpy
import json
import importlib
import numpy as np
import pandas as pd
import random
from zipfile import ZipFile
from gym import spaces, Env
# import wandb
import yaml
import pickle
import shutil
from datetime import datetime
import re
import copy
import argparse
from sklearn.model_selection import train_test_split
from mpc_policy import get_mpc_controller


class LinearDecayEpsilonMPC(d3rlpy.online.explorers.LinearDecayEpsilonGreedy):

    def __init__(
        self,
        mpc_policy,
        action_scaler,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.1,
        duration: int = 1000000,
    ):
        super().__init__(start_epsilon=start_epsilon,
                         end_epsilon=end_epsilon,
                         duration=duration)
        self.action_scaler = action_scaler
        self.mpc_policy = mpc_policy

    def sample(self, algo, x, step):
        greedy_actions = algo.predict(x)
        mpc_actions = self.mpc_policy.predict(x)
        greedy_actions = np.nan_to_num(greedy_actions, nan=0.0, posinf=0.0, neginf=0.0)
        mpc_actions = np.nan_to_num(mpc_actions, nan=0.0, posinf=0.0, neginf=0.0)
        is_random = np.random.random(x.shape[0]) < self.compute_epsilon(step)
        actions = np.where(is_random, mpc_actions, greedy_actions)
        if np.isnan(actions).any():
            print("actions: ", actions)
            print("x: ", x)
            sys.exit(0)
        return actions



dir_loc = os.path.dirname(os.path.relpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--algo', type=str, help='algorithm', default="CQL")
parser.add_argument('--online', action='store_true', help="online learning")
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
    # env_name = config_dict['env_name']
    env_name = args.env
    env_lib = importlib.import_module(f"examples.{env_name}.template_env")
    
    model_name = config_dict['model_name']
    dense_reward = config_dict['dense_reward']
    debug_mode = config_dict['debug_mode']
    
    data_loc = os.environ['AMLT_DATA_DIR'] if args.amlt else dir_loc
    dataset_dir = os.path.join(data_loc, 'datasets', env_name)
    use_gpu = (False if args.device < 0 else args.device)

    # for offlineRL online learning
    online_training = args.online # config_dict['online_training']
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
    BATCH_SIZE = config_dict["batch_size"]
    scaler = config_dict['scaler']
    action_scaler = config_dict['action_scaler']
    reward_scaler = config_dict['reward_scaler']
    evaluate_on_environment = config_dict['evaluate_on_environment']
    default_loc = os.path.join(data_loc, config_dict['default_loc'], env_name)
    training_dataset_loc = os.path.join(data_loc, config_dict['training_dataset_loc'], env_name)

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

    # if not online_training:
    #     with open(training_dataset_loc, 'rb') as handle:
    #         training_dataset_pkl = pickle.load(handle)
    #     with open(eval_dataset_loc, 'rb') as handle:
    #         eval_dataset_pkl = pickle.load(handle)

    if online_training:
        algo_names = ['CQL', 'PLAS', 'PLASWithPerturbation', 'BEAR', 'SAC', 'BCQ', 'CRR', 'AWAC', 'DDPG', 'TD3', 'COMBO', 'MOPO', 'BC']
        default_loc += '_ONLINE'
    else:
        algo_names = ['BC', 'CQL', 'PLAS', 'PLASWithPerturbation', 'BEAR', 'SAC', 'BCQ', 'CRR', 'AWAC', 'DDPG', 'TD3', 'TD3PlusBC', 'COMBO', 'MOPO']

    for seed in seeds:
        # set random seeds in random module, numpy module and PyTorch module.
        d3rlpy.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        env = env_lib.get_env()
        
        dataset = None
        for _iter in range(args.iter+1):
            _training_dataset_loc = os.path.join(training_dataset_loc, str(_iter))
            for file_loc in os.listdir(_training_dataset_loc):
                data_loc = os.path.join(_training_dataset_loc, file_loc)
                print("loading data in", data_loc)  
                _dataset = d3rlpy.dataset.MDPDataset.load(data_loc)
                if dataset is None: dataset = _dataset
                else: dataset.extend(_dataset)
        assert dataset is not None, "training data is empty"
        train_episodes, test_episodes = train_test_split(dataset.episodes)
        feeded_episodes = train_episodes
        eval_feeded_episodes = test_episodes
        print(dataset.actions[:10])
        
        for algo_name in args.algo.split(","):
            prev_evaluate_on_environment_scorer = float('-inf')
            prev_continuous_action_diff_scorer = float('inf')
            global ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER
            ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER = float('-inf')

            if args.online:
                # scaler = d3rlpy.preprocessing.MinMaxScaler(maximum=env.max_observation, minimum=env.min_observation)
                # action_scaler = d3rlpy.preprocessing.MinMaxActionScaler(maximum=env.max_actions, minimum=env.min_actions)
                scaler = None
                action_scaler = None
                reward_scaler = d3rlpy.preprocessing.MinMaxRewardScaler(minimum=-4.0, maximum=0.0)
            else:
                scaler = d3rlpy.preprocessing.MinMaxScaler(dataset)
                action_scaler = d3rlpy.preprocessing.MinMaxActionScaler(dataset)
                reward_scaler = d3rlpy.preprocessing.MinMaxRewardScaler(minimum=-4.0, maximum=0.0)
            encoder_factory = d3rlpy.models.encoders.VectorEncoderFactory(activation="relu", dropout_rate=0.0, hidden_units=[128,128], use_dense=True)
            optim_factory=d3rlpy.models.optimizers.AdamFactory(optim_cls='Adam', betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)

            if algo_name == 'CQL':
                curr_algo = d3rlpy.algos.CQL(q_func_factory='mean', use_gpu=use_gpu, 
                                             batch_size = BATCH_SIZE, scaler = scaler, 
                                             actor_learning_rate=0.001, 
                                             critic_learning_rate=0.003, 
                                             temp_learning_rate=0.001, 
                                             alpha_learning_rate=0.0,
                                             conservative_weight=5.0,
                                             action_scaler=action_scaler, 
                                             reward_scaler=reward_scaler,
                                             actor_encoder_factory=encoder_factory, 
                                             critic_encoder_factory=encoder_factory,
                                             actor_optim_factory=optim_factory,
                                             critic_optim_factory=optim_factory)
            elif algo_name == 'PLAS':
                curr_algo = d3rlpy.algos.PLAS(q_func_factory='qr', use_gpu=use_gpu, batch_size = BATCH_SIZE, scaler = scaler, action_scaler=action_scaler, reward_scaler=reward_scaler) # use Quantile Regression Q function, default was 'mean'
            elif algo_name == 'PLASWithPerturbation':
                curr_algo = d3rlpy.algos.PLASWithPerturbation(q_func_factory='qr', use_gpu=use_gpu, batch_size = BATCH_SIZE, scaler = scaler, action_scaler=action_scaler, reward_scaler=reward_scaler) # use Quantile Regression Q function, default was 'mean'
            elif algo_name == 'DDPG':
                curr_algo = d3rlpy.algos.DDPG(use_gpu=use_gpu, batch_size = BATCH_SIZE, scaler = scaler, action_scaler=action_scaler, reward_scaler=reward_scaler)
            elif algo_name == 'BC':
                curr_algo = d3rlpy.algos.BC(use_gpu=use_gpu, batch_size = BATCH_SIZE, scaler = scaler, action_scaler=action_scaler, reward_scaler=reward_scaler)
            elif algo_name == 'TD3':
                curr_algo = d3rlpy.algos.TD3(use_gpu=use_gpu, batch_size = BATCH_SIZE, scaler = scaler, action_scaler=action_scaler, reward_scaler=reward_scaler)
            elif algo_name == 'TD3PlusBC':
                curr_algo = d3rlpy.algos.TD3PlusBC(use_gpu=use_gpu, batch_size = BATCH_SIZE, scaler = scaler, action_scaler=action_scaler, reward_scaler=reward_scaler)
            elif algo_name == 'BEAR':
                curr_algo = d3rlpy.algos.BEAR(use_gpu=use_gpu, batch_size = BATCH_SIZE, scaler = scaler, action_scaler=action_scaler, reward_scaler=reward_scaler)
            elif algo_name == 'SAC':
                curr_algo = d3rlpy.algos.SAC(use_gpu=use_gpu, batch_size = BATCH_SIZE, 
                                             actor_learning_rate=0.0001, 
                                             critic_learning_rate=0.0003, 
                                             temp_learning_rate=0.001, 
                                             alpha_learning_rate=0.001,
                                             scaler = scaler, action_scaler=action_scaler, 
                                             reward_scaler=reward_scaler,
                                             actor_encoder_factory=encoder_factory, 
                                             critic_encoder_factory=encoder_factory,
                                             actor_optim_factory=optim_factory,
                                             critic_optim_factory=optim_factory)
            elif algo_name == 'BCQ':
                curr_algo = d3rlpy.algos.BCQ(use_gpu=use_gpu, batch_size = BATCH_SIZE, scaler = scaler, action_scaler=action_scaler, reward_scaler=reward_scaler)
            elif algo_name == 'CRR':
                curr_algo = d3rlpy.algos.CRR(use_gpu=use_gpu, batch_size = BATCH_SIZE, scaler = scaler, action_scaler=action_scaler, reward_scaler=reward_scaler)
            elif algo_name == 'AWR':
                curr_algo = d3rlpy.algos.AWR(use_gpu=use_gpu, batch_size = BATCH_SIZE, scaler = scaler, action_scaler=action_scaler, reward_scaler=reward_scaler)
            elif algo_name == 'AWAC':
                curr_algo = d3rlpy.algos.AWAC(use_gpu=use_gpu, batch_size = BATCH_SIZE, 
                                              scaler = scaler, 
                                              action_scaler=action_scaler, 
                                              reward_scaler=reward_scaler,
                                              actor_encoder_factory=encoder_factory, 
                                              critic_encoder_factory=encoder_factory,
                                              actor_optim_factory=optim_factory,
                                              critic_optim_factory=optim_factory)
            elif algo_name == 'COMBO':
                dynamics = 1
            #     dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=use_gpu)
            #     curr_algo = d3rlpy.algos.COMBO(use_gpu=use_gpu)
            elif algo_name == 'MOPO':
                dynamics = 1
            #     dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=use_gpu)
            #     curr_algo = d3rlpy.algos.MOPO(use_gpu=use_gpu)
            else:
                raise Exception("algo_name is invalid!", algo_name)
            # print(dataset_name, env.action_space.shape, env.observation_space.shape, len(dataset.episodes), np.ceil(len(dataset.episodes)*0.01))
            
            logdir = f"{default_loc}/{args.exp}_{seed}/"
            actual_dir = logdir+'/'+algo_name
            if os.path.exists(actual_dir): shutil.rmtree(actual_dir)
            os.makedirs(logdir, exist_ok=True)
            if algo_name in ['COMBO', 'MOPO']:
                scorers={
                    'observation_error': d3rlpy.metrics.scorer.dynamics_observation_prediction_error_scorer,
                    'reward_error': d3rlpy.metrics.scorer.dynamics_reward_prediction_error_scorer,
                    'variance': d3rlpy.metrics.scorer.dynamics_prediction_variance_scorer,
                }
                # train dynamics model first
                dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=use_gpu, scaler = scaler, action_scaler=action_scaler, reward_scaler=reward_scaler)
                dynamics.fit(feeded_episodes,
                    eval_episodes=eval_feeded_episodes,
                    n_epochs=DYNAMICS_N_EPOCHS,
                    logdir=logdir,
                    scorers=scorers)
                # dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics.from_json("d3rlpy_logs/CSTR/random_42/ProbabilisticEnsembleDynamics_20230301130632/params.json")
                # dynamics.load_model("d3rlpy_logs/CSTR/random_42/ProbabilisticEnsembleDynamics_20230301130632/model_375000.pt")
                if algo_name == 'COMBO':
                    curr_algo = d3rlpy.algos.COMBO(dynamics=dynamics, use_gpu=use_gpu, 
                                                   scaler = scaler, 
                                                   action_scaler=action_scaler, 
                                                   reward_scaler=reward_scaler,
                                                   actor_encoder_factory=encoder_factory, 
                                                   critic_encoder_factory=encoder_factory)
                elif algo_name == 'MOPO':
                    curr_algo = d3rlpy.algos.MOPO(dynamics=dynamics, 
                                                  use_gpu=use_gpu, 
                                                  scaler = scaler, 
                                                  action_scaler=action_scaler, 
                                                  reward_scaler=reward_scaler,
                                                  actor_encoder_factory=encoder_factory, 
                                                  critic_encoder_factory=encoder_factory)
                else:
                    raise Exception("algo_name is invalid!")
            # --------- Model Based Algorithms leverages the probablistic ensemble dynamics model to generate new dynamics data with uncertainty penalties.  --------- 
            
            if algo_name == 'BC':
                scorers = {
                    'continuous_action_diff_scorer': d3rlpy.metrics.scorer.continuous_action_diff_scorer,
                }
            elif algo_name == 'AWR':
                scorers = {
                    'td_error': d3rlpy.metrics.scorer.td_error_scorer,
                    'value_scale': d3rlpy.metrics.scorer.average_value_estimation_scorer,
                    'discounted_sum_of_advantage_scorer': d3rlpy.metrics.scorer.discounted_sum_of_advantage_scorer,
                    # 'value_estimation_std_scorer': d3rlpy.metrics.scorer.value_estimation_std_scorer,
                    'initial_state_value_estimation_scorer': d3rlpy.metrics.scorer.initial_state_value_estimation_scorer,
                    'continuous_action_diff_scorer': d3rlpy.metrics.scorer.continuous_action_diff_scorer,
                }
            else:
                scorers = {
                    'td_error': d3rlpy.metrics.scorer.td_error_scorer,
                    'value_scale': d3rlpy.metrics.scorer.average_value_estimation_scorer,
                    'discounted_sum_of_advantage_scorer': d3rlpy.metrics.scorer.discounted_sum_of_advantage_scorer,
                    'value_estimation_std_scorer': d3rlpy.metrics.scorer.value_estimation_std_scorer,
                    'initial_state_value_estimation_scorer': d3rlpy.metrics.scorer.initial_state_value_estimation_scorer,
                    'continuous_action_diff_scorer': d3rlpy.metrics.scorer.continuous_action_diff_scorer,
                }
                
            if evaluate_on_environment:
                scorers['evaluate_on_environment_scorer'] = d3rlpy.metrics.scorer.evaluate_on_environment(env)
            
            if online_training:
                def online_saving_callback(algo, epoch, total_step):
                    mean_env_ret = d3rlpy.metrics.evaluate_on_environment(env, n_trials=2, epsilon=0.0)(algo)
                    global ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER
                    if mean_env_ret > ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER:
                        ONLINE_PREV_EVALUATE_ON_ENVIRONMENT_SCORER = mean_env_ret
                        curr_algo.save_model(os.path.join(logdir, 'best_env.pt'))


                mpc_policy = get_mpc_controller(env)
                # explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy(start_epsilon=explorer_start_epsilon, end_epsilon=explorer_end_epsilon, duration=explorer_duration)
                explorer = LinearDecayEpsilonMPC(mpc_policy, action_scaler, start_epsilon=explorer_start_epsilon, end_epsilon=explorer_end_epsilon, duration=explorer_duration)
                buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=buffer_maxlen, env=env)
                
                curr_algo.fit_online(env, buffer, explorer=explorer, # you don't need this with probablistic policy algorithms
                    eval_env=env,
                    eval_epsilon=0.0,
                    n_steps=N_EPOCHS*n_steps_per_epoch,
                    n_steps_per_epoch=n_steps_per_epoch,
                    update_interval=online_update_interval,
                    random_steps=online_random_steps,
                    save_interval=online_save_interval,
                    with_timestamp=False,
                    tensorboard_dir=actual_dir+'/tensorboard',
                    logdir=actual_dir,
                    callback=online_saving_callback)
            else:
                best_ckpt = os.path.join(actual_dir, 'best.pt')
                if args.iter > 0 and os.path.exists(best_ckpt): curr_algo.load_model(best_ckpt)
                for epoch, metrics in curr_algo.fitter(feeded_episodes, eval_episodes=eval_feeded_episodes, n_epochs=N_EPOCHS, with_timestamp=False, logdir=logdir, scorers=scorers):
                    # done = False
                    # observation = env.reset()
                    # step_cnt = 0
                    # while not done:
                    #     observation = np.array(observation).reshape((1, -1))
                    #     action = curr_algo.predict(observation)[0]
                    #     observation, reward, done, _ = env.step(action)
                    #     print(step_cnt, "Action: ", action, "reward: ", reward)
                    #     # print("prev. obs.: ", env.previous_observation, "cur. obs.: ", observation)
                    #     step_cnt += 1
                    if evaluate_on_environment:
                        if metrics['evaluate_on_environment_scorer'] > prev_evaluate_on_environment_scorer:
                            prev_evaluate_on_environment_scorer = metrics['evaluate_on_environment_scorer']
                            curr_algo.save_model(os.path.join(actual_dir, 'best_evaluate_on_environment_scorer.pt'))
                    if metrics['continuous_action_diff_scorer'] < prev_continuous_action_diff_scorer:
                        prev_continuous_action_diff_scorer = metrics['continuous_action_diff_scorer']
                        curr_algo.save_model(os.path.join(actual_dir, 'best_continuous_action_diff_scorer.pt'))
                    if evaluate_on_environment:
                        shutil.copyfile(os.path.join(actual_dir, 'best_evaluate_on_environment_scorer.pt'), os.path.join(actual_dir, 'best.pt'))
                    else:
                        shutil.copyfile(os.path.join(actual_dir, 'best_continuous_action_diff_scorer.pt'), os.path.join(actual_dir, 'best.pt'))
            # wandb_run.finish()
