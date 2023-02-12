import gym

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import do_mpc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from examples.batch_reactor.template_model import template_model
from examples.batch_reactor.template_mpc import template_mpc
from examples.batch_reactor.template_simulator import template_simulator

""" User settings: """
show_animation = False
store_results = False

"""
Get configured do-mpc modules:
"""

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""

X_s_0 = 1.0 # This is the initial concentration inside the tank [mol/l]
S_s_0 = 0.5 # This is the controlled variable [mol/l]
P_s_0 = 0.0 #[C]
V_s_0 = 120.0 #[C]
x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0])

mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()
for k in range(150):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)


# Reward Function
# 1. distance between current state and steady state
# 2. distance between current state and the previous state
# 3. distance between current action and the previous action

class BatchReactorEnv(gym.Env):
    def __init__(self, 
                 model, 
                 simulator, 
                 estimator,
                 steady_observation=None):
        self.model = model # model defines ODE/PDE of the dynamics
        self.simulator = simulator # taking model as an input, simulate its execution given the current state and controls
        self.estimator = estimator # state feature estimator, in case state is not fully observable, currently, simply return state of the simulator
        self.action_dim = len(self.simulator.u0.keys())
        self.observation_dim = len(self.simulator.x0.keys())
        # All scalers are one of the following:
        # None - No scaler
        # sklearn.preprocessing.MinMaxScaler
        # sklearn.preprocessing.StandardScaler
        self.action_scaler = None
        self.reward_scaler = None
        self.observation_scaler = None
        # A steady observation specifies the ideal state that the system should stay close to
        self.steady_observation = steady_observation
        self.min_observation, self.max_observation = self.observation_scaler.min_values, self.observation_scaler.max_values
        self.min_actions, self.max_actions = self.action_scaler.min_values, self.action_scaler.max_values
        if self.model.model_type == "discrete":
                self.action_space = gym.spaces.Discrete(self.action_dim)
        else:
                self.action_space = gym.spaces.Box(low=self.min_actions, high=self.max_actions, shape=(self.action_dim,))
        self.observation_space = gym.spaces.Box(low=self.min_observation, high=self.max_observation, shape=(self.observation_dim,))
        self.cur_step = 0
        self.episode_len = 50
            
    def get_reward(self, state, action):
        return 0.0

    def step(self, action):
        if self.action_scaler is not None: action = self.action_scaler.inverse_transform(action)
        state_next = self.simulator.make_step(action)
        state_next = self.estimator.make_step(state_next)
        if self.observation_scaler is not None: state_next = self.observation_scaler.transform(state_next)
        reward = self.get_reward(self.state, state_next, action)
        self.cur_step += 1
        done = (self.cur_step >= self.episode_len)
        info = {}
        self.state = state_next
        return self.state, reward, done, info

    def reset(self, init_state=np.array([1.0, 0.5, 0.0, 120.0])):
        self.simulator.reset_history()
        self.simulator.x0 = init_state
        self.estimator.x0 = init_state
        self.state = init_state
        return self.state