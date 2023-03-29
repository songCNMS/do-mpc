import yaml
import os


def load_yaml_config(file_loc):
    with open(file_loc) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def generate_model(config, loc):
    model_str = f"""
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_model(symvar_type='SX'):
    model_type = "{config['model']['model_type']}" # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)
"""
    if "constants" in config['model']:
        model_str += "    # certain parameters \n"
        for key, val in config['model']['constants'].items():
            model_str += f"    {key} = {val} \n"
    
    if "state_variables" in config["model"]:
        model_str += "    # States struct (optimization variables) \n"
        for key, val in config['model']['state_variables'].items():
            shape = val.get("shape", "(1,)")
            if shape == "(1,)": model_str += f"    {key} = model.set_variable('_x', '{key}') \n"
            else: model_str += f"    {key} = model.set_variable('_x', '{key}', shape={shape}) \n"
            
    if "control_variables" in config["model"]:
        model_str += "    # Input struct (optimization variables) \n"
        for key, val in config['model']['control_variables'].items():
            shape = val.get("shape", "(1,)")
            if shape == "(1,)": model_str += f"    {key} = model.set_variable('_u', '{key}') \n"
            else: model_str += f"    {key} = model.set_variable('_u', '{key}', shape={shape}) \n"
    
    if "user_defined_parameters" in config["model"]:
        for val in config['model']['user_defined_parameters']:
            model_str += f"    {val} = model.set_variable('_p', '{val}') \n"
    
    # T_dif = model.set_expression(expr_name='T_dif', expr=T_R-T_K)
    if "aux_variables" in config["model"]:
        for key, val in config['model']['aux_variables'].items():
            if val.get("is_explicit", False): model_str += f"    {key}=model.set_expression(expr_name='{key}', expr={val['expr']}) \n"
            else: model_str += f"    {key}={val['expr']} \n"
            
    if "state_variables" in config["model"]:
        for key, val in config['model']['state_variables'].items():
            assert "rhs" in val, "rhs must be given for each state variable"
            model_str += f"    model.set_rhs('{key}', {val['rhs']}) \n"

    model_str += "    model.setup() \n"
    model_str += "    return model"
    with open(loc, "w") as f:
        f.write(model_str)
    return


def generate_simulator(config, loc):
    simulator_str = f"""
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_simulator(model):
    simulator = do_mpc.simulator.Simulator(model)
"""
    simulator_str += f"    params_simulator = {str(config['simulator']['parameters'])} \n"
    simulator_str += f"    simulator.set_param(**params_simulator) \n"
    simulator_str += f"    p_num = simulator.get_p_template() \n"
    
    p_values = config["mpc"]["uncertainties"]
    for key, vals in p_values.items():
        simulator_str += f"    {key}_values = {vals} \n"
        simulator_str += f"    p_num['{key}'] = {vals[0]} \n"
    simulator_str += f"    def p_fun(t_now):\n"
    for key, vals in p_values.items():
        simulator_str += f"        p_num['{key}']=np.random.choice({key}_values) \n"
    simulator_str += f"        return p_num \n"
        
    simulator_str += f"    simulator.set_p_fun(p_fun)\n"
    simulator_str += f"    simulator.setup()\n"
    simulator_str += f"    return simulator"
    
    simulator_str += generate_reward_function(config)
    
    with open(loc, "w") as f:
        f.write(simulator_str)
    return


def generate_mpc(config, loc):
    mpc_str = f"""
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_mpc(model):
    mpc = do_mpc.controller.MPC(model)
"""
    mpc_str += f"    setup_mpc = {str(config['mpc']['setup'])} \n"
    mpc_str += f"    mpc.set_param(**setup_mpc) \n"
    p_values = config["mpc"]["uncertainties"]
    for key, vals in p_values.items():
        mpc_str += f"    {key}_values = np.array({vals}) \n"
    
    mterm = ""
    if "step_reward" in config["reward"]:
        for key, vals in config["reward"]["step_reward"].items():
            expr = vals['expr'].replace(key, f"model.x['{key}']")
            mterm = f"{-vals['coef']}*{expr}"
            
    if "terminal_reward" in config["reward"]:
        lterm = ""
        for key, vals in config["reward"]["terminal_reward"].items():
            expr = vals['expr'].replace(key, f"model.x['{key}']")
            lterm += f"{-vals['coef']}*{expr}"
    objective_str = ""
    if mterm != "": 
        mpc_str += f"    mterm={mterm} \n"
        objective_str += "mterm=mterm, "
    if lterm != "": 
        objective_str += "lterm=lterm"
        mpc_str += f"    lterm={lterm} \n"
    mpc_str += f"    mpc.set_objective({objective_str}) \n"
    
    if "input_reward" in config["reward"]:
        rterm = ",".join([f"{key}={val}" for key, val in config["reward"]["input_reward"].items()])
        mpc_str += f"    mpc.set_rterm({rterm}) \n"

    if "scaling" in config["mpc"]:
        for key, val in config["mpc"]["scaling"].items():
            type_str = ("_x" if key in config["model"]["state_variables"] else "_u")
            mpc_str += f"    mpc.scaling['{type_str}', '{key}'] = {val} \n"

    if "bounds" in config["mpc"]:
        for key, vals in config["mpc"]["bounds"].items():
            type_str = ("_x" if key in config["model"]["state_variables"] else "_u")
            for bound_str in ["lower", "upper"]:
                if bound_str in vals:
                    if bound_str == "lower" or ("soft" not in vals) or (not vals["soft"]):
                        type_str = ("_x" if key in config["model"]["state_variables"] else "_u")
                        mpc_str += f"    mpc.bounds['{bound_str}', '{type_str}', '{key}'] = {vals[bound_str]} \n"
                    else: 
                        coef = vals.get("coef", 1.0)
                        type_str = ("model.x" if key in config["model"]["state_variables"] else "model.u")
                        mpc_str += f"    mpc.set_nl_cons('{key}', {type_str}['{key}'], ub={vals[bound_str]}, soft_constraint=True, penalty_term_cons={coef}) \n"
    
    if "uncertainities" in config["mpc"]:
        for key, val in config["mpc"]["uncertainities"].items():
            mpc_str += f"    {key}_values = np.array({val}) \n"
        uncertainities_str = ",".join([f"{key}={key}_values" for key in config["mpc"]["uncertainities"].keys()])
        mpc_str += f"    mpc.set_uncertainty_values({uncertainities_str}) \n"
           
    mpc_str += f"    mpc.setup()\n"
    mpc_str += f"    return mpc"
    with open(loc, "w") as f:
        f.write(mpc_str)
    return
    

def generate_reward_function(config):
    reward_str = """
    
def reward_function(cur_step, simulator_data):
"""
    if "step_reward" in config["reward"]:
        for key, vals in config["reward"]["step_reward"].items():
            expr_str = vals["expr"].replace(key, f"simulator_data['_x', '{key}'][-1, 0]")
            reward_str += f"    reward = {vals['coef']}*{expr_str} \n"
    if "bounds" in config["mpc"]:
        for key, vals in config["mpc"]["bounds"].items():
            type_str = ("_x" if key in config["model"]["state_variables"] else "_u")
            bound_str = "upper"
            if bound_str in vals:
                if vals.get("soft", False):
                    coef = vals.get("coef", 1.0)
                    reward_str += f"    reward -= {coef}*max(0, simulator_data['{type_str}', '{key}'][-1, 0]-{vals['upper']}) \n"
    if "input_reward" in config["reward"]:
        reward_str += f"    if cur_step > 0:\n"
        for key, val in config["reward"]["input_reward"].items():
            reward_str += f"        reward -= {val}*(simulator_data['_u', '{key}'][-1, 0]-simulator_data['_u', '{key}'][-2, 0])**2 \n"
    reward_str += "    return reward"
    return reward_str


def generate_env(config, loc):
    env_str = """
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
"""
    x_list = []
    min_observation_list, max_observation_list = [], []
    min_action_list, max_action_list = [], []
    init_min_observation_list, init_max_observation_list = [], []
    
    state_variables_config = config['model']['state_variables']
    variables_bounds_config = config['mpc']['bounds']
    state_variables = list(state_variables_config.keys())
    control_variables = list(config['model']['control_variables'].keys())
    
    for key in state_variables:
        val = state_variables_config[key]
        env_str += f"    {key}_0={val['init_val']} \n"
        x_list.append(key+"_0")
        init_min_observation_list.append(val['init_val_lower'])
        init_max_observation_list.append(val['init_val_upper'])
        val = variables_bounds_config[key]
        min_observation_list.append(val['lower'])
        max_observation_list.append(val['upper'])
    
    for key in control_variables:
        val = variables_bounds_config[key]
        min_action_list.append(val['lower'])
        max_action_list.append(val['upper'])
            
    x_list_str = ",".join(x_list)
    env_str += f"""
    x0 = np.array([{x_list_str}])
    simulator.x0 = x0
    estimator.x0 = x0
    mpc.x0 = x0
"""
    min_observation_str = ",".join([str(v) for v in min_observation_list])
    max_observation_str = ",".join([str(v) for v in max_observation_list])
    min_action_str = ",".join([str(v) for v in min_action_list])
    max_action_str = ",".join([str(v) for v in max_action_list])
    init_min_observation_str = ",".join([str(v) for v in init_min_observation_list])
    init_max_observation_str = ",".join([str(v) for v in init_max_observation_list])

    env_str += f"""
    min_observation = np.array([{min_observation_str}])
    max_observation = np.array([{max_observation_str}])
    min_actions = np.array([{min_action_str}])
    max_actions = np.array([{max_action_str}])
    
    def init_obs_space(seed):
        init_min_observation = np.array([{init_min_observation_str}])
        init_max_observation = np.array([{init_max_observation_str}])
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
"""
    with open(loc, "w") as f:
        f.write(env_str)
    return



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='config file', required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.file)
    
    model_name = config["model"]["model_name"]
    dir_loc = f"examples/{model_name}"
    os.makedirs(dir_loc, exist_ok=True)
    
    generate_model(config, os.path.join(dir_loc, "template_model.py"))
    generate_simulator(config, os.path.join(dir_loc, "template_simulator.py"))
    generate_mpc(config, os.path.join(dir_loc, "template_mpc.py"))
    generate_env(config, os.path.join(dir_loc, "template_env.py"))
    os.system(f"touch {os.path.join(dir_loc, '__init__.py')}")