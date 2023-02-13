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
parser.add_argument("--steps", type=int, help='total samples', default="1000")

if __name__ == '__main__':
    args = parser.parse_args()
    data_loc = os.environ['AMLT_DATA_DIR'] if args.amlt else dir_loc
    dataset_dir = os.path.join(data_loc, 'datasets')

    env = get_env()
    policy = get_mpc_controller(env, noise=0.2)
    
    num_steps = args.steps
    # prepare experience replay buffer
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=num_steps, env=env)

    # start data collection
    policy.collect(env, buffer, n_steps=num_steps)

    # export as MDPDataset
    dataset = buffer.to_mdp_dataset()

    # save MDPDataset
    os.makedirs(dataset_dir, exist_ok=True)
    dataset.dump(f"{dataset_dir}/mpc_policy_batch_reactor_dataset_{num_steps}.h5")