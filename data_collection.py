import sys
sys.path.insert(0, "/mnt/lesong/do-mpc")
import importlib
import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import do_mpc
import d3rlpy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from examples.batch_reactor.template_model import template_model
from examples.batch_reactor.template_mpc import template_mpc
from examples.batch_reactor.template_simulator import template_simulator
from mpc_policy import get_mpc_controller, get_noisy_rl_policy
from algo_evaluation import OfflineRLModel

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument("--start_episodes", type=int, help='total samples', default="0")
parser.add_argument("--end_episodes", type=int, help='total samples', default="1000")
parser.add_argument("--seed", type=int, help='seed', default="13")
parser.add_argument('--env', type=str, help='env. name', default="CSTR")
parser.add_argument('--iter', type=int, help='iteration', default=0)
parser.add_argument('--algo', type=str, help='algo. used to collect data', default="CQL")
parser.add_argument('--model', type=str, help='model loc. used to collect data', default="random_42")


dir_loc = os.path.dirname(os.path.relpath(__file__))

if __name__ == '__main__':
    args = parser.parse_args()
    data_loc = os.environ['AMLT_DATA_DIR'] if args.amlt else dir_loc
    dataset_dir = os.path.join(data_loc, 'datasets', args.env, str(args.iter))
    env_lib = importlib.import_module(f"examples.{args.env}.template_env")
    env = env_lib.get_env()
    
    episodes_per_batch = 100
    os.makedirs(dataset_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for batch_idx in range(args.start_episodes, args.end_episodes, episodes_per_batch):
        noise = rng.choice([0.0, 0.05, 0.1])
        if args.algo == "MPC": policy = get_mpc_controller(env, noise=noise)
        else:
            model_loc = os.path.join("d3rlpy_logs", args.env, args.model)
            base_policy = OfflineRLModel(args.algo, model_loc)
            policy = get_noisy_rl_policy(env, base_policy, noise)
        # prepare experience replay buffer
        buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=episodes_per_batch*env.episode_len, env=env)
        # start data collection
        policy.collect(env, buffer, n_steps=episodes_per_batch*env.episode_len)
        # export as MDPDataset
        dataset = buffer.to_mdp_dataset()
        # save MDPDataset
        dataset.dump(f"{dataset_dir}/{args.algo}_dataset_{noise}_{batch_idx}.h5")