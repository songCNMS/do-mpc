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

from gym_env_wrapper import get_env
from mpc_policy import get_mpc_controller

dir_loc = os.path.dirname(os.path.relpath(__file__))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument("--num_episodes", type=int, help='total samples', default="1000")


if __name__ == '__main__':
    args = parser.parse_args()
    data_loc = os.environ['AMLT_DATA_DIR'] if args.amlt else dir_loc
    dataset_dir = os.path.join(data_loc, 'datasets')

    env = get_env()
    episodes_per_batch = 100
    num_episodes = args.num_episodes
    os.makedirs(dataset_dir, exist_ok=True)
    for batch_idx in range(0, num_episodes, episodes_per_batch):
        env.reset()
        noise = np.random.choice([0.1,0.2,0.3,0.4,0.5])
        policy = get_mpc_controller(env, noise=noise)
        # prepare experience replay buffer
        buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=episodes_per_batch*env.episode_len, env=env)
        # start data collection
        policy.collect(env, buffer, n_steps=episodes_per_batch*env.episode_len)
        # export as MDPDataset
        dataset = buffer.to_mdp_dataset()
        # save MDPDataset
        dataset.dump(f"{dataset_dir}/mpc_policy_batch_reactor_dataset_{noise}_{batch_idx}.h5")