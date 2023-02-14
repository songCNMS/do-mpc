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


# Reward Function
# 1. distance between current state and steady state
# 2. distance between current state and the previous state
# 3. distance between current action and the previous action

class BatchReactorEnv(gym.Env):
    def __init__(self,
                 model, 
                 simulator, 
                 estimator,
                 min_observation,
                 max_observation,
                 min_actions,
                 max_actions,
                 seed=0,
                 observation_scaler=None,
                 action_scaler=None,
                 reward_scaler=None,
                 steady_observation=None):
        self.model = model # model defines ODE/PDE of the dynamics
        self.simulator = simulator # taking model as an input, simulate its execution given the current state and controls
        self.estimator = estimator # state feature estimator, in case state is not fully observable, currently, simply return state of the simulator
        self.min_observation, self.max_observation = min_observation, max_observation
        self.min_actions, self.max_actions = min_actions, max_actions
        self.action_dim = len(min_actions)
        self.observation_dim = len(min_observation)
        self.observation_name = self.estimator.x0.keys()
        self.action_name = self.estimator.u0.keys()[1:]
        # All scalers are those defined in d3rlpy:
        self.action_scaler = action_scaler
        self.reward_scaler = reward_scaler
        self.observation_scaler = observation_scaler
        # A steady observation specifies the ideal state that the system should stay close to
        self.steady_observation = steady_observation
        self.steady_action = [0.0]
        
        if self.model.model_type == "discrete":
                self.action_space = gym.spaces.Discrete(self.action_dim)
        else:
                self.action_space = gym.spaces.Box(low=self.min_actions, high=self.max_actions, shape=(len(self.min_actions),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.min_observation, high=self.max_observation, dtype=np.float32)
        self.cur_step = 0
        self.episode_len = 100
        self.error_reward = -1000.0
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        
            
    def get_reward(self, state, state_next, action):
        # print(self.simulator.data["_x", "P_s"])
        reward = self.simulator.data["_x", "P_s"][-1, 0]
        if self.reward_scaler is not None: reward = self.reward_scaler.transform(reward)
        return reward

    def step(self, action):
        # print("ori action: ", action)
        action = action.reshape(self.model._u.shape)
        if self.action_scaler is not None: action = self.action_scaler.inverse_transform(action)
        state_next = self.simulator.make_step(action)
        state_next = self.estimator.make_step(state_next).flatten()
        if self.observation_scaler is not None: state_next = self.observation_scaler.transform(state_next)
        reward = self.get_reward(self.state, state_next, action)
        self.cur_step += 1
        done = (self.cur_step >= self.episode_len)
        info = {}
        self.state = state_next
        # print(self.cur_step, self.state, reward, done, info)
        return self.state, reward, done, info

    def reset(self, init_state=None, seed=None):
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        if init_state is None: init_state = self.observation_space.sample()
        self.simulator.reset_history()
        self.simulator.x0 = init_state
        self.estimator.x0 = init_state
        self.state = init_state
        self.cur_step = 0
        return self.state


def get_env():
    model = template_model()
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
    simulator.x0 = x0
    estimator.x0 = x0
    
    min_observation = np.array([0.0, -0.01, 0.0, 0.0])
    max_observation = np.array([3.7, 2.0, 3.0, 150.0])
    min_actions = np.array([0.0])
    max_actions = np.array([0.3])
    
    env = BatchReactorEnv(model, simulator, estimator,
                          min_observation, max_observation,
                          min_actions, max_actions,
                          steady_observation=x0)
    env.reset()
    return env



if __name__ == '__main__':

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
    for k in range(10):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next).flatten()
        print("step: ", k, u0, x0)