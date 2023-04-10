
import yaml


def eval_bool(val_str):
    if val_str in ["True", "true", "1", "on"]:
        return True
    else:
        return False
    
def model2yaml(wd, model_config):
    for i in range(wd["num_constants"]):
        param = wd[f"constant_{i}"]
        model_config["model"]["constants"][param["parameter_name"]] = float(param["parameter_value"])

    for i in range(wd["num_state_variables"]):
        param = wd[f"state_variable_{i}"]
        shape = param["parameter_shape"]
        model_config["model"]["state_variables"][param["parameter_name"]] = {
            "init_val": (
                float(param["parameter_init_value"])
                if shape == "(1,)"
                else [float(param["parameter_init_value"])]
            ),
            "init_val_lower": (
                float(param["parameter_lower"])
                if shape == "(1,)"
                else [float(param["parameter_lower"])]
            ),
            "init_val_upper": (
                param["parameter_upper"]
                if shape == "(1,)"
                else [param["parameter_upper"]]
            ),
            "rhs": param["parameter_rhs"],
            "shape": shape,
            "scaling": float(param["parameter_scaling"]),
        }

    for i in range(wd["num_controls"]):
        param = wd[f"control_{i}"]
        model_config["model"]["control_variables"][param["parameter_name"]] = {
            "shape": param["parameter_shape"]
        }

    model_config["model"]["user_defined_parameters"] = [
        wd[f"parameter_{i}"]["parameter_name"] for i in range(wd["num_parameters"])
    ]

    for i in range(wd["num_aux_variables"]):
        param = wd[f"aux_variable_{i}"]
        model_config["model"]["aux_variables"][param["parameter_name"]] = {
            "expr": param["parameter_expr"]
        }
    return model_config


def simulator2yaml(wd, model_config):
    model_config["simulator"] = {"parameters": {}}
    for i in range(wd["num_simulator_parameters"]):
        param = wd[f"simulator_parameter_{i}"]
        type_eval = eval(param["parameter_type"])
        if type_eval is bool:
            model_config["simulator"]["parameters"][
                param["parameter_name"]
            ] = eval_bool(param["parameter_value"])
        else:
            model_config["simulator"]["parameters"][
                param["parameter_name"]
            ] = type_eval(param["parameter_value"])
    return model_config


def reward2yaml(wd, model_config):
    model_config["reward"] = {
        "step_reward": {},
        "terminal_reward": {},
        "input_reward": {}
    }
    for i in range(wd["num_state_variables"]):
        param = wd[f"state_variable_{i}"]
        param_step = wd[f"state_step_{param['parameter_name']}"]
        param_terminal = wd[f"state_terminal_{param['parameter_name']}"]
        model_config["reward"]["step_reward"][param['parameter_name']] = {
            "expr": param_step["parameter_expr"],
            "coef": float(param_step["parameter_coef"]),
        }
        model_config["reward"]["terminal_reward"][param["parameter_name"]] = {
            "expr": param_terminal["parameter_expr"],
            "coef": float(param_terminal["parameter_coef"]),
        }
    for i in range(wd["num_controls"]):
        param = wd[f"control_{i}"]
        param_reward = wd[f"input_{param['parameter_name']}"]
        model_config["reward"]["input_reward"][param['parameter_name']] = {
            param["parameter_name"]: float(param_reward["parameter_coef"])
        }
    return model_config


def mpc2yaml(wd, model_config):
    model_config["mpc"] = {"setup": {}}
    for i in range(wd["num_MPC_parameters"]):
        param = wd[f"MPC_parameter_{i}"]
        type_eval = eval(param["parameter_type"])
        if type_eval is bool:
            model_config["mpc"]["setup"][param["parameter_name"]] = eval_bool(
                param["parameter_value"]
            )
        else:
            model_config["mpc"]["setup"][param["parameter_name"]] = type_eval(
                param["parameter_value"]
            )

    sv_names = [wd[f"state_variable_{i}"]["parameter_name"] for i in range(wd["num_state_variables"])]
    in_names = [
        wd[f"control_{i}"]["parameter_name"] for i in range(wd["num_controls"])
    ]
    model_config["mpc"]["scaling"] = {}
    model_config["mpc"]["bounds"] = {}
    model_config["mpc"]["uncertainties"] = {}
    for var_name in sv_names + in_names:
        param = wd[f"MPC_{var_name}"]
        model_config["mpc"]["scaling"][var_name] = param["parameter_scaling"]
        model_config["mpc"]["bounds"][var_name] = {}
        model_config["mpc"]["bounds"][var_name]["lower"] = float(param["parameter_lower"])
        model_config["mpc"]["bounds"][var_name]["upper"] = float(param["parameter_upper"])
        model_config["mpc"]["bounds"][var_name]["soft"] = eval_bool(param["parameter_soft"])

    num_parameters = wd["num_parameters"]
    for i in range(num_parameters):
        param = wd[f"parameter_{i}"]
        para_val = eval(param["parameter_values"])
        model_config["mpc"]["uncertainties"][param["parameter_name"]] = para_val
    return model_config


def estimator2yaml(wd, model_config):
    model_config["estimator"] = {"type": wd["est_type"], "parameters": {}}
    for i in range(wd["num_estimator_parameters"]):
        param = wd[f"estimator_parameter_{i}"]
        type_eval = eval(param["parameter_type"])
        if type_eval is bool:
            model_config["estimator"]["parameters"][
                param["parameter_name"]
            ] = eval_bool(param["parameter_value"])
        else:
            model_config["estimator"]["parameters"][
                param["parameter_name"]
            ] = type_eval(param["parameter_value"])
    return model_config


def save2yaml(wd, yaml_file=None):
    model_config = {}
    model_config["model"] = {
        "model_name": wd["model_name"],
        "model_type": wd["model_type"],
        "constants": {},
        "state_variables": {},
        "control_variables": {},
        "user_defined_parameters": [],
        "aux_variables": {},
    }

    model_config = model2yaml(wd, model_config)
    model_config = simulator2yaml(wd, model_config)
    model_config = reward2yaml(wd, model_config)
    model_config = mpc2yaml(wd, model_config)
    model_config = estimator2yaml(wd, model_config)
    if yaml_file is not None:
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(model_config, f)
    return model_config