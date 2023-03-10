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
    
    if "aux_variables" in config["model"]:
        for key, val in config['model']['aux_variables'].items():
            model_str += f"    {key} = {val} \n"
            
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
    
    p_values = config["mpc"]["uncertainities"]
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
    p_values = config["mpc"]["uncertainities"]
    for key, vals in p_values.items():
        mpc_str += f"    {key}_values = np.array({vals}) \n"
    
    mterm = ""
    if "step_reward" in config["reward"]:
        for key, vals in config["reward"]["step_reward"].items():
            expr = vals['expr'].replace(key, f"model.x['{key}']")
            mterm += f"-{vals['coef']}*{expr}"
            
    if "terminal_reward" in config["reward"]:
        lterm = ""
        for key, vals in config["reward"]["terminal_reward"].items():
            expr = vals['expr'].replace(key, f"model.x['{key}']")
            lterm += f"-{vals['coef']}*{expr}"
    objective_str = ""
    if mterm != "": 
        mpc_str += f"    mterm={mterm} \n"
        objective_str += f"mterm={mterm}, "
    if lterm != "": 
        objective_str += f"lterm={lterm}"
        mpc_str += f"    lterm={lterm} \n"
    mpc_str += f"    mpc.set_objective({objective_str}) \n"
    
    if "input_reward" in config["reward"]:
        rterm = ",".join([f"{key}={val}" for key, val in config["reward"]["input_reward"].items()])
        mpc_str += f"    mpc.set_rterm({rterm}) \n"

    if "bounds" in config["mpc"]:
        for key, vals in config["mpc"]["bounds"].items():
            type_str = ("_x" if key in config["model"]["state_variables"] else "_u")
            for bound_str in ["lower", "upper"]:
                if bound_str in vals:
                    if ("soft" not in vals) or (not vals["soft"]):
                        mpc_str += f"    mpc.bounds['{bound_str}', '{type_str}', '{key}'] = {vals[bound_str]} \n"
                    else: 
                        coef = vals.get("coef", 1.0)
                        short_bound_str=("ub" if bound_str=="upper" else "lb")
                        mpc_str += f"     mpc.set_nl_cons('{key}', {type_str}['{key}'], {short_bound_str}={vals[bound_str]}, soft_constraint=True, penalty_term_cons={coef}) \n"
    
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
    pass


if __name__ == "__main__":

    config = load_yaml_config("model_config.yaml")
    print(config["model"]["user_defined_parameters"])
    model_name = config["model"]["model_name"]
    dir_loc = f"examples/{model_name}"
    os.makedirs(dir_loc, exist_ok=True)
    
    generate_model(config, os.path.join(dir_loc, "template_model.py"))
    generate_simulator(config, os.path.join(dir_loc, "template_simulator.py"))
    generate_mpc(config, os.path.join(dir_loc, "template_mpc.py"))