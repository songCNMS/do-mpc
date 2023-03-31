
import sys
sys.path.append('../../')
import do_mpc
import numpy as np
import gym
from mpc_env.gym_env_wrapper import ControlEnv
from .template_model import template_model
from .template_mpc import template_mpc
from .template_simulator import template_simulator, reward_function


def get_env():
    model = template_model()
    simulator = template_simulator(model)
    estimator = do_mpc.estimator.StateFeedback(model)
    mpc = template_mpc(model)
    C_a_0=[0.8] 
    C_b_0=[0.5] 
    T_R_0=[134.14] 
    T_K_0=[130.0] 

    x0 = np.array([C_a_0,C_b_0,T_R_0,T_K_0])
    simulator.x0 = x0
    estimator.x0 = x0
    mpc.x0 = x0

    min_observation = np.array([0.1,0.1,50.0,50.0])
    max_observation = np.array([2.0,2.0,140.0,140.0])
    min_actions = np.array([5.0,-8500.0])
    max_actions = np.array([100.0,0.0])
    
    def init_obs_space(seed):
        init_min_observation = np.array([[0.1],[0.1],[50.0],[50.0]])
        init_max_observation = np.array([[2.0],[2.0],[150.0],[140.0]])
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
