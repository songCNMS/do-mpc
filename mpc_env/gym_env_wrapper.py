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



# TODO: historical observations in the state
class ControlEnv(gym.Env):
    def __init__(self,
                 model, 
                 simulator, 
                 estimator,
                 mpc,
                 min_observation,
                 max_observation,
                 min_actions,
                 max_actions,
                 reward_function,
                 init_obs_space=None,
                 seed=0,
                 steady_observation=None,
                 steady_action=None,
                 error_reward=-1000,
                 obs_refactor=None):
        self.model = model # model defines ODE/PDE of the dynamics
        self.simulator = simulator # taking model as an input, simulate its execution given the current state and controls
        self.estimator = estimator # state feature estimator, in case state is not fully observable, currently, simply return state of the simulator
        self.mpc = mpc
        self.min_observation, self.max_observation = min_observation, max_observation
        self.min_actions, self.max_actions = min_actions, max_actions
        self.action_dim = len(min_actions)
        self.observation_dim = len(min_observation)
        self.observation_name = self.estimator.x0.keys()
        self.action_name = self.estimator.u0.keys()[1:]
        # A steady observation specifies the ideal state that the system should stay close to
        self.steady_observation = steady_observation
        self.steady_action = (np.array([0.0]*self.action_dim) if steady_action is None else steady_action)
        self.reward_function = reward_function
        
        if self.model.model_type == "discrete":
                self.action_space = gym.spaces.Discrete(self.action_dim)
        else:
                self.action_space = gym.spaces.Box(low=self.min_actions, high=self.max_actions, shape=(len(self.min_actions),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=self.min_observation, high=self.max_observation, dtype=np.float32)
        self.cur_step = 0
        self.episode_len = self.mpc.n_horizon
        self._max_episode_steps = self.episode_len
        self.error_reward = error_reward
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.obs_refactor = obs_refactor
        self.init_state_sampler = (init_obs_space(seed) if init_obs_space is not None else None)
            
    def get_reward(self):
        return self.reward_function(self.cur_step, self.simulator.data)
        
    def step(self, action):
        action = action.reshape(self.model._u.shape)
        state_next = self.simulator.make_step(action)
        state_next = self.estimator.make_step(state_next).flatten()
        reward = self.get_reward()
        self.cur_step += 1
        done = (self.cur_step >= self.episode_len)
        info = {}
        self.state = state_next
        # print(self.cur_step, self.state, reward, done, info)
        if np.isnan(self.state).any() or np.isnan(reward):
            self.state = np.nan_to_num(self.state, nan=0.0, posinf=0.0, neginf=0.0)
            done = True
        return self.state, reward, done, info

    def reset(self, init_state=None, seed=None):
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            if self.init_state_sampler is not None: self.init_state_sampler.seed(seed)
        self.simulator.reset_history()
        self.estimator.reset_history()
        if init_state is None: 
            if self.init_state_sampler is None: init_state = self.observation_space.sample()
            else: init_state = self.init_state_sampler.sample()
            if self.obs_refactor is not None: init_state = self.obs_refactor(init_state)
        self.simulator.x0 = init_state
        self.estimator.x0 = init_state
        self.mpc.x0 = init_state
        self.state = init_state
        self.cur_step = 0
        # self.mpc.set_initial_guess()
        return self.state


def get_batch_reactor_env():
    from examples.batch_reactor.template_model import template_model
    from examples.batch_reactor.template_mpc import template_mpc
    from examples.batch_reactor.template_simulator import template_simulator, reward_function
    model = template_model()
    simulator = template_simulator(model)
    estimator = do_mpc.estimator.StateFeedback(model)
    mpc = template_mpc(model)
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
    mpc.x0 = x0
    
    min_observation = np.array([0.0, -0.01, 0.0, 0.0])
    max_observation = np.array([3.7, 2.0, 3.0, 150.0])
    min_actions = np.array([0.0])
    max_actions = np.array([0.3])
    
    env = ControlEnv(model, simulator, estimator, mpc,
                     min_observation, max_observation,
                     min_actions, max_actions,
                     reward_function,
                     steady_observation=x0,
                     error_reward=-1000)
    env.reset(init_state=x0)
    return env


def get_CSTR_env():
    from examples.CSTR.template_model import template_model
    from examples.CSTR.template_mpc import template_mpc
    from examples.CSTR.template_simulator import template_simulator, reward_function
    model = template_model()
    simulator = template_simulator(model)
    estimator = do_mpc.estimator.StateFeedback(model)
    mpc = template_mpc(model)
    """
    Set initial state
    """
    C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
    C_b_0 = 0.5 # This is the controlled variable [mol/l]
    T_R_0 = 134.14 #[C]
    T_K_0 = 130.0 #[C]
    x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)
    simulator.x0 = x0
    estimator.x0 = x0
    mpc.x0 = x0

    min_observation = np.array([0.1, 0.1, 50.0, 50.0])
    max_observation = np.array([2.0, 2.0, 150.0, 140.0])
    min_actions = np.array([5.0, -8500.0])
    max_actions = np.array([100.0, 0.0])
    
    def init_obs_space(seed):
        init_min_observation = np.array([0.6, 0.3, 125, 125])
        init_max_observation = np.array([1.0, 0.8, 140, 135])
        observation_space = gym.spaces.Box(low=init_min_observation, high=init_max_observation, dtype=np.float32)
        observation_space.seed(seed)
        return observation_space
    
    env = ControlEnv(model, simulator, estimator, mpc,
                    min_observation, max_observation,
                    min_actions, max_actions,
                    reward_function,
                    init_obs_space=init_obs_space,
                    steady_observation=x0,
                    error_reward=-1000)
    env.reset(init_state=x0)
    return env


def get_IPR_env():
    from examples.industrial_poly.template_model import template_model
    from examples.industrial_poly.template_mpc import template_mpc
    from examples.industrial_poly.template_simulator import template_simulator, reward_function
    model = template_model()
    simulator = template_simulator(model)
    estimator = do_mpc.estimator.StateFeedback(model)
    mpc = template_mpc(model)
    # Set the initial state of the controller and simulator:
    delH_R_real = 950.0
    c_pR = 5.0
    def IPR_obs_refactor(obs):
        p = [delH_R_real, c_pR]
        obs[-1] = obs[1]*p[0]/((obs[0]+obs[1]+obs[2])*p[1]) + obs[3]
        return obs
    # x0 is a property of the simulator - we obtain it and set values.
    x0 = simulator.x0
    x0['m_W'] = 10000.0
    x0['m_A'] = 853.0
    x0['m_P'] = 26.5
    x0['T_R'] = 90.0 + 273.15
    x0['T_S'] = 90.0 + 273.15
    x0['Tout_M'] = 90.0 + 273.15
    x0['T_EK'] = 35.0 + 273.15
    x0['Tout_AWT'] = 35.0 + 273.15
    x0['accum_monom'] = 300.0
    T_adiab = x0['m_A']*delH_R_real/((x0['m_W'] + x0['m_A'] + x0['m_P']) * c_pR) + x0['T_R']
    x0['T_adiab'] = T_adiab
    temp_range = 2.0
    simulator.x0 = x0
    estimator.x0 = x0
    mpc.x0 = x0

    min_observation = np.array([0.0, 0.0, 26.0, 363.15-temp_range, 298.0, 298.0, 288.0, 288.0, 0.0, -10000.0])
    max_observation = np.array([15000.0, 1000.0, 36.0, 363.15+1.2*temp_range, 400.0, 400.0, 400.0, 400.0, 30000.0, 100000.0])
    min_actions = np.array([0.0, 333.15, 333.15])
    max_actions = np.array([3.0e4, 373.15, 373.15])
    
    init_state = np.array([10000.0, 853.0, 26.5, 90.0 + 273.15, 90.0 + 273.15, 90.0 + 273.15, 35.0 + 273.15, 35.0 + 273.15, 300.0, 359.144])
    def init_obs_space(seed):
        init_min_observation = init_state*0.95
        init_max_observation = init_state*1.05
        init_min_observation = IPR_obs_refactor(init_min_observation)
        init_max_observation = IPR_obs_refactor(init_max_observation)
        observation_space = gym.spaces.Box(low=init_min_observation, high=init_max_observation, dtype=np.float32)
        observation_space.seed(seed)
        return observation_space
    
    env = ControlEnv(model, simulator, estimator, mpc,
                     min_observation, max_observation,
                     min_actions, max_actions,
                     reward_function,
                     init_obs_space=init_obs_space,
                     steady_observation=init_state,
                     error_reward=-1000,
                     obs_refactor=IPR_obs_refactor)
    env.reset()
    return env


def get_env(env_name):
    if env_name == "CSTR": env = get_CSTR_env()
    elif env_name == "batch_reactor": env = get_batch_reactor_env()
    elif env_name == "IPR": env = get_IPR_env()
    else: raise Exception("unknown environment", env_name)
    return env


if __name__ == '__main__':

    """ User settings: """
    show_animation = False
    store_results = False
    """
    Get configured do-mpc modules:
    """
    env = get_IPR_env()
    mpc = env.mpc
    simulator = env.simulator
    estimator = env.estimator
    """
    Set initial state
    """
    delH_R_real = 950.0
    c_pR = 5.0
    x0 = simulator.x0
    x0['m_W'] = 10000.0
    x0['m_A'] = 853.0
    x0['m_P'] = 26.5
    x0['T_R'] = 90.0 + 273.15
    x0['T_S'] = 90.0 + 273.15
    x0['Tout_M'] = 90.0 + 273.15
    x0['T_EK'] = 35.0 + 273.15
    x0['Tout_AWT'] = 35.0 + 273.15
    x0['accum_monom'] = 300.0
    x0['T_adiab'] = x0['m_A']*delH_R_real/((x0['m_W'] + x0['m_A'] + x0['m_P']) * c_pR) + x0['T_R']

    env.reset()
    mpc.set_initial_guess()
    for k in range(10):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next).flatten()
        print("step: ", k, u0, x0)
        