from django.shortcuts import render, redirect, reverse
from django.http import QueryDict
from vis.utility import save2yaml
# from vis.models import *


def step1(request):
    init_context = {
        "model_name": request.session.get("model_name", "Model Name"),
        "model_type": request.session.get("model_type", "continuous"),
        "num_constants": request.session.get("num_constants", 2),
        "num_state_variables": request.session.get("num_state_variables", 2),
        "num_controls": request.session.get("num_controls", 2),
        "num_parameters": request.session.get("num_parameters", 2),
        "num_aux_variables": request.session.get("num_aux_variables", 2)
    }
    if request.method == 'POST':
        # Handle form submission
        model_name = request.POST.get('model name', init_context["model_name"])
        model_type = request.POST.get('model type', init_context["model_type"])
        num_constants = int(request.POST.get('constants', init_context["num_constants"]))
        num_state_variables = int(request.POST.get('state variables', init_context["num_state_variables"]))
        num_controls = int(request.POST.get('control variables', init_context["num_controls"]))
        num_parameters = int(request.POST.get('user defined parameters', init_context["num_parameters"]))
        num_aux_variables = int(request.POST.get('auxiliary variables', init_context["num_aux_variables"]))
        request.session['model_name'] = model_name
        request.session['model_type'] = model_type
        request.session['num_constants'] = num_constants
        request.session['num_state_variables'] = num_state_variables
        request.session['num_controls'] = num_controls
        request.session['num_parameters'] = num_parameters
        request.session['num_aux_variables'] = num_aux_variables
        url = reverse('step2')
        return redirect(url)
    else:
        # Display form for inputting a number
        return render(request, 'step1.html', init_context)

def step2(request):
    init_context = {
        "model_name": request.session.get("model_name", "Model Name"),
        "model_type": request.session.get("model_type", "continuous"),
        "num_constants": request.session.get("num_constants", 2),
        "num_state_variables": request.session.get("num_state_variables", 2),
        "num_controls": request.session.get("num_controls", 2),
        "num_parameters": request.session.get("num_parameters", 2),
        "num_aux_variables": request.session.get("num_aux_variables", 2)
    }
    
    if request.method == 'POST':
        for i in range(init_context["num_constants"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=request.POST.get(f"constant {i} name", f"constant_{i}"), 
                        parameter_value=request.POST.get(f"constant {i} value", "value"))
            q_dict.update(data)
            request.session[f"constant_{i}"] = q_dict
        for i in range(init_context["num_state_variables"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=request.POST.get(f"state variable {i} name", f"state_variable_name_{i}"), 
                        parameter_init_value=request.POST.get(f"state variable {i} init value", "0"),
                        parameter_lower=request.POST.get(f"state variable {i} lower bound", "0"),
                        parameter_upper=request.POST.get(f"state variable {i} upper bound", "10000"),
                        parameter_scaling=request.POST.get(f"state variable {i} scaling", "1"),
                        parameter_shape=request.POST.get(f"state variable {i} shape", "(1,)"),
                        parameter_rhs=request.POST.get(f"state variable {i} RHS", "RHS"),
                        parameter_soft=request.POST.get(f"state variable {i} soft", "off")
                        )
            q_dict.update(data)
            request.session[f"state_variable_{i}"] = q_dict
        
        for i in range(init_context["num_controls"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=request.POST.get(f"control {i} name", f"control_{i}"), 
                        parameter_shape=request.POST.get(f"control {i} shape", "(1,)"))
            q_dict.update(data)
            request.session[f"control_{i}"] = q_dict
        
        for i in range(init_context["num_parameters"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=request.POST.get(f"parameter {i} name", f"parameter_{i}"), 
                        parameter_values=request.POST.get(f"parameter {i} values", "[0.0]"))
            q_dict.update(data)
            request.session[f"parameter_{i}"] = q_dict
            
        for i in range(init_context["num_aux_variables"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=request.POST.get(f"auxiliary variable {i} name", f"aux_variable_{i}"), 
                        parameter_expr=request.POST.get(f"auxiliary variable {i} expr", "(1,)"))
            q_dict.update(data)
            request.session[f"aux_variable_{i}"] = q_dict
        
        url = reverse('step3')
        return redirect(url)
    else:
        constant_list = []
        for i in range(init_context["num_constants"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=f"constant_{i}", 
                        parameter_value="value")
            q_dict.update(data)
            q_dict = request.session.get(f"constant_{i}", q_dict)
            constant_list.append(q_dict)
        
        state_variable_list = []
        for i in range(init_context["num_state_variables"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=f"state_variable_name_{i}", 
                        parameter_init_value="0",
                        parameter_lower=0,
                        parameter_upper=10000,
                        parameter_scaling=1,
                        parameter_shape="(1,)",
                        parameter_rhs="RHS",
                        parameter_soft="off"
                        )
            q_dict.update(data)
            q_dict = request.session.get(f"state_variable_{i}", q_dict)
            state_variable_list.append(q_dict)
        
        control_list = []
        for i in range(init_context["num_controls"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=f"control_{i}", 
                        parameter_shape="(1,)")
            q_dict.update(data)
            q_dict = request.session.get(f"control_{i}", q_dict)
            control_list.append(q_dict)
        
        parameter_list = []
        for i in range(init_context["num_parameters"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=f"parameter_{i}", 
                        parameter_values="[0.0]")
            q_dict.update(data)
            q_dict = request.session.get(f"parameter_{i}", q_dict)
            parameter_list.append(q_dict)
        
        aux_variable_list = []
        for i in range(init_context["num_aux_variables"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=f"aux_variable_{i}", 
                        parameter_expr="(1,)")
            q_dict.update(data)
            q_dict = request.session.get(f"aux_variable_{i}", q_dict)
            aux_variable_list.append(q_dict)
        init_context.update({
            "constants": constant_list,
            "state_variables": state_variable_list,
            "controls": control_list,
            "parameters": parameter_list,
            "aux_variables": aux_variable_list})
        return render(request, 'step2.html', init_context)

    
def step3(request):
    init_context = {
        "num_simulator_parameters": request.session.get("num_simulator_parameters", 2)
    }
    if request.method == 'POST':
        num_simulator_parameters = int(request.POST.get('num simulator parameters', init_context["num_simulator_parameters"]))
        request.session['num_simulator_parameters'] = num_simulator_parameters
        init_context['num_simulator_parameters'] = num_simulator_parameters
        for i in range(init_context["num_simulator_parameters"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=request.POST.get(f"simulator parameter {i} name", f"simulator_parameter_{i}"), 
                        parameter_value=request.POST.get(f"simulator parameter {i} value", "value"),
                        parameter_type=request.POST.get(f"simulator parameter {i} type", "str"))
            q_dict.update(data)
            request.session[f"simulator_parameter_{i}"] = q_dict
        if "next-step" in request.POST:
            url = reverse('step4')
            return redirect(url)
    simulator_parameters_config = []
    for i in range(init_context["num_simulator_parameters"]):
        q_dict = QueryDict('', mutable=True)
        init_parameter = dict(parameter_idx=i,
                              parameter_name=f"simulator_parameter_{i}", 
                              parameter_value="value",
                              parameter_type="str")
        q_dict.update(init_parameter)
        simulator_parameter = request.session.get(f"simulator_parameter_{i}", q_dict)
        simulator_parameters_config.append(simulator_parameter)
    init_context["simulator_parameters"] = simulator_parameters_config
    return render(request, 'step3.html', init_context)


def step4(request):
    init_context = {
        "num_MPC_parameters": request.session.get("num_MPC_parameters", 2)
    }
    
    sv_names, cv_names = [], []
    for i in range(request.session["num_state_variables"]): sv_names.append(request.session[f"state_variable_{i}"]["parameter_name"])
    for i in range(request.session["num_controls"]): cv_names.append(request.session[f"control_{i}"]["parameter_name"])
    
    if request.method == 'POST':
        num_MPC_parameters = int(request.POST.get('num MPC parameters', init_context["num_MPC_parameters"]))
        request.session['num_MPC_parameters'] = num_MPC_parameters
        init_context['num_MPC_parameters'] = num_MPC_parameters
        for i in range(init_context["num_MPC_parameters"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=request.POST.get(f"MPC parameter {i} name", f"MPC_parameter_{i}"), 
                        parameter_value=request.POST.get(f"MPC parameter {i} value", "value"),
                        parameter_type=request.POST.get(f"MPC parameter {i} type", "str"))
            q_dict.update(data)
            request.session[f"MPC_parameter_{i}"] = q_dict
        # store mpc configs
        for param in sv_names + cv_names:
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_name=param, 
                        parameter_lower=request.POST.get(f"{param} lower bound", "0"),
                        parameter_upper=request.POST.get(f"{param} upper bound", "10000"),
                        parameter_scaling=request.POST.get(f"{param} scaling", "1.0"),
                        parameter_soft=request.POST.get(f"{param} soft", "off")
                        )
            q_dict.update(data)
            request.session[f"MPC_{param}"] = q_dict
        if "next-step" in request.POST:
            url = reverse('step5')
            return redirect(url)
    MPC_parameters_config = []
    for i in range(init_context["num_MPC_parameters"]):
        q_dict = QueryDict('', mutable=True)
        init_parameter = dict(parameter_idx=i,
                              parameter_name=f"MPC_parameter_{i}", 
                              parameter_value="value",
                              parameter_type="str")
        q_dict.update(init_parameter)
        MPC_parameter = request.session.get(f"MPC_parameter_{i}", q_dict)
        MPC_parameters_config.append(MPC_parameter)
    init_context["MPC_parameters"] = MPC_parameters_config
    init_context["sv_names"] = sv_names
    init_context["cv_names"] = cv_names
    mpc_configs = []
    for param in sv_names + cv_names:
        init_q_dict = QueryDict('', mutable=True)
        init_data = dict(parameter_name=param, 
                    parameter_type=("state variable" if param in sv_names else "control variable"),
                    parameter_lower="0",
                    parameter_upper="10000",
                    parameter_scaling="1.0",
                    parameter_soft="off"
                    )
        init_q_dict.update(init_data)
        q_dict = request.session.get(f"MPC_{param}", init_q_dict)
        mpc_configs.append(q_dict)
    init_context["MPC_configs"] = mpc_configs
    return render(request, 'step4.html', init_context)


def step5(request):
    init_context = {
        "num_estimator_parameters": request.session.get("num_estimator_parameters", 2),
        "est_type": request.session.get("estimator type", "StateFeedBack")
    }
    
    if request.method == 'POST':
        num_estimator_parameters = int(request.POST.get('num estimator parameters', init_context["num_estimator_parameters"]))
        request.session['num_estimator_parameters'] = num_estimator_parameters
        init_context['num_estimator_parameters'] = num_estimator_parameters
        init_context["est_type"] = request.POST.get('estimator type', init_context["est_type"])
        request.session["est_type"] = init_context["est_type"]
        for i in range(init_context["num_estimator_parameters"]):
            q_dict = QueryDict('', mutable=True)
            data = dict(parameter_idx=i,
                        parameter_name=request.POST.get(f"estimator parameter {i} name", f"estimator_parameter_{i}"), 
                        parameter_value=request.POST.get(f"estimator parameter {i} value", "value"),
                        parameter_type=request.POST.get(f"estimator parameter {i} type", "str"))
            q_dict.update(data)
            request.session[f"estimator_parameter_{i}"] = q_dict
        if "next-step" in request.POST:
            url = reverse('step6')
            return redirect(url)
    estimator_parameters_config = []
    for i in range(init_context["num_estimator_parameters"]):
        q_dict = QueryDict('', mutable=True)
        init_parameter = dict(parameter_idx=i,
                              parameter_name=f"estimator_parameter_{i}", 
                              parameter_value="value",
                              parameter_type="str")
        q_dict.update(init_parameter)
        estimator_parameter = request.session.get(f"estimator_parameter_{i}", q_dict)
        estimator_parameters_config.append(estimator_parameter)
    init_context["estimator_parameters"] = estimator_parameters_config
    init_context["est_type"] = request.session.get("est_type", "StateFeedBack")
    return render(request, 'step5.html', init_context)



def step6(request):
    sv_names, cv_names = [], []
    for i in range(request.session["num_state_variables"]): sv_names.append(request.session[f"state_variable_{i}"]["parameter_name"])
    for i in range(request.session["num_controls"]): cv_names.append(request.session[f"control_{i}"]["parameter_name"])
    
    if request.method == 'POST':
        for param in sv_names:
            q_dict1 = QueryDict('', mutable=True)
            data = dict(parameter_name=param,
                        parameter_type="state variable",
                        parameter_expr=request.POST.get(f"state step {param} expr", param),
                        parameter_coef=request.POST.get(f"state step {param} coef", "0.0"),
                        )
            q_dict1.update(data)
            request.session[f"state_step_{param}"] = q_dict1
            
            q_dict2 = QueryDict('', mutable=True)
            data = dict(parameter_name=param,
                        parameter_type="state variable",
                        parameter_expr=request.POST.get(f"state terminal {param} expr", param),
                        parameter_coef=request.POST.get(f"state terminal {param} coef", "0.0"),
                        )
            q_dict2.update(data)
            request.session[f"state_terminal_{param}"] = q_dict2
        
        for param in cv_names:
            q_dict3 = QueryDict('', mutable=True)
            data = dict(parameter_name=param,
                        parameter_type="control variable", 
                        parameter_coef=request.POST.get(f"control {param} coef", "0.0"),
                        )
            q_dict3.update(data)
            request.session[f"input_{param}"] = q_dict3
        url = reverse('step7')
        return redirect(url)
    state_step_rewards = []
    state_terminal_rewards = []
    input_rewards = []
    for param in sv_names:
        ss_q_dict = QueryDict('', mutable=True)
        st_q_dict = QueryDict('', mutable=True)
        init_data = dict(parameter_name=param, 
                         parameter_type="state variable",
                         parameter_expr=param,
                         parameter_coef="0"
                      )
        ss_q_dict.update(init_data)
        ss_q_dict = request.session.get(f"state_step_{param}", ss_q_dict)
        state_step_rewards.append(ss_q_dict)
        st_q_dict.update(init_data)
        st_q_dict = request.session.get(f"state_terminal_{param}", st_q_dict)
        state_terminal_rewards.append(st_q_dict)
    for param in cv_names:
        in_q_dict = QueryDict('', mutable=True)
        init_data = dict(parameter_name=param, 
                         parameter_type="control variable",
                         parameter_coef="0"
                        )
        in_q_dict.update(init_data)
        in_q_dict = request.session.get(f"input_{param}", in_q_dict)
        input_rewards.append(in_q_dict)
        
    init_context = {
        "state_step_rewards": state_step_rewards,
        "state_terminal_rewards": state_terminal_rewards,
        "input_rewards": input_rewards
    }
    return render(request, 'step6.html', init_context)

import yaml

def step7(request):
    yaml_config = save2yaml(request.session)
    yaml_config = yaml.dump(yaml_config)
    print(yaml_config)
    return render(request, 'step7.html', {"yaml_config": yaml_config})



