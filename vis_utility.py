from ipywidgets import *
import yaml


def eval_bool(val_str):
    if val_str in ["True", "true", "1"]:
        return True
    else:
        return False


def layout_generator(header, children, num_cols=2, col_width=200, row_height=40):
    num_rows = len(children) // num_cols
    aux_cols = 1
    if len(children) % num_cols:
        footer = children[-1]
        aux_cols += 1
    else:
        footer = None

    grid = GridspecLayout(
        num_rows + aux_cols,
        num_cols,
        width=f"{col_width*num_cols}px",
        height=f"{row_height*(num_rows+aux_cols)}px",
        merge=True,
    )
    grid[0, :] = header
    for row in range(num_rows):
        for col in range(num_cols):
            grid[row + 1, col] = children[row * num_cols + col]
    if footer:
        for col in range(num_cols):
            if num_rows * num_cols + col < len(children):
                grid[-1, col] = children[num_rows * num_cols + col]
    return grid


def var_num_grid_generator(default_wd=None):
    header = Button(
        description="Model Specification",
        layout=Layout(width="auto", grid_area="header"),
        style=ButtonStyle(button_color="lightblue"),
    )
    model_name = Text(
        value=(
            "Model Name" if default_wd is None else default_wd["model"]["model_name"]
        ),
        placeholder="Type something",
        description="Model Name:",
        disabled=False,
    )

    model_type = Dropdown(
        options=["continuous", "discrete"],
        value=(
            "continuous" if default_wd is None else default_wd["model"]["model_type"]
        ),
        description="Model Type:",
        disabled=False,
    )

    num_constants = BoundedIntText(
        value=(
            2
            if default_wd is None
            else len(default_wd["model"].get("constants", {}).keys())
        ),
        min=0,
        step=1,
        description="Num. Constants:",
        style={"description_width": "initial"},
        disabled=False,
    )

    num_state_variables = BoundedIntText(
        value=(
            2
            if default_wd is None
            else len(default_wd["model"].get("state_variables", {}).keys())
        ),
        min=0,
        step=1,
        description="Num. State Variables:",
        style={"description_width": "initial"},
        disabled=False,
    )

    num_control_variables = BoundedIntText(
        value=(
            2
            if default_wd is None
            else len(default_wd["model"].get("control_variables", {}).keys())
        ),
        min=0,
        step=1,
        description="Num. Control Variables:",
        style={"description_width": "initial"},
        disabled=False,
    )

    num_parameters = BoundedIntText(
        value=(
            2
            if default_wd is None
            else len(default_wd["model"].get("user_defined_parameters", []))
        ),
        min=0,
        step=1,
        description="Num. User Defined Parameters",
        style={"description_width": "initial"},
        disabled=False,
    )

    num_aux_variables = BoundedIntText(
        value=(
            2
            if default_wd is None
            else len(default_wd["model"].get("aux_variables", {}).keys())
        ),
        min=0,
        step=1,
        description="Num. Auxiliary Variables:",
        style={"description_width": "initial"},
        disabled=False,
    )

    widget_dict = {
        "model_name": model_name,
        "model_type": model_type,
        "num_constants": num_constants,
        "num_state_variables": num_state_variables,
        "num_control_variables": num_control_variables,
        "num_parameters": num_parameters,
        "num_aux_variables": num_aux_variables,
    }

    grid = layout_generator(header, list(widget_dict.values()), col_width=400)
    return grid, widget_dict


def constants_grid_generator(num_constants, default_wd=None):
    widget_dict = {}
    grid = None
    if num_constants > 0:
        constant_children = []
        constant_names = []
        constant_values = []
        constants_header = Button(
            description="Constants Definition",
            layout=Layout(width="auto", grid_area="header"),
            style=ButtonStyle(button_color="lightblue"),
        )
        constant_names = (
            [f"X_{i}" for i in range(num_constants)]
            if default_wd is None
            else list(default_wd["model"].get("constants", {}).keys())
        )
        for i in range(num_constants):
            constant_name = f"X_{i}" if i >= len(constant_names) else constant_names[i]
            constant_def = Text(
                value=constant_name,
                placeholder=constant_name,
                description=f"{i+1}-th Constant Name:",
                style={"description_width": "initial"},
                disabled=False,
            )

            constant_val = FloatText(
                value=(
                    0.0
                    if default_wd is None
                    else default_wd["model"]["constants"].get(constant_names[i], 0.0)
                ),
                description=f"Value:",
                disabled=False,
            )
            constant_children.extend([constant_def, constant_val])
            widget_dict.update(
                {f"constant_{i}_def": constant_def, f"constant_{i}_val": constant_val}
            )
            constant_names.append(constant_def)
            constant_values.append(constant_val)

        grid = layout_generator(constants_header, constant_children, col_width=400)
    return grid, widget_dict


def state_variables_grid_generator(num_state_variables, default_wd=None):
    widget_dict = {}
    grid = None
    if num_state_variables > 0:
        state_variables_children = []
        state_variables_names = []
        state_variables_init_values = []
        state_variables_lower_bounds = []
        state_variables_upper_bounds = []
        state_variables_rhs = []
        state_variables_shape = []
        state_variables_scaling = []
        state_variables_soft = []

        state_variables_header = Button(
            description="State Variables Definition",
            layout=Layout(width="auto", grid_area="header"),
            style=ButtonStyle(button_color="lightblue"),
        )
        state_variables = (
            [f"X_{i}" for i in range(num_state_variables)]
            if default_wd is None
            else list(default_wd["model"].get("state_variables", {}).keys())
        )
        for i in range(num_state_variables):
            var_name = state_variables[i] if i < len(state_variables) else f"X_{i}"
            shape = (
                "(1,)"
                if default_wd is None
                else default_wd["model"]["state_variables"]
                .get(var_name, {})
                .get("shape", "(1,)")
            )
            sv_def = Text(
                value=var_name,
                placeholder=var_name,
                description=f"{i+1}-th State Variable Name:",
                style={"description_width": "initial"},
                disabled=False,
            )
            if shape == "(1,)":
                val = (
                    0.0
                    if default_wd is None
                    else default_wd["model"]["state_variables"]
                    .get(var_name, {})
                    .get("init_val", 0.0)
                )
            else:
                val = (
                    0.0
                    if default_wd is None
                    else default_wd["model"]["state_variables"]
                    .get(var_name, {})
                    .get("init_val", [0.0])[0]
                )
            sv_init_val = FloatText(
                value=val, description=f"Init Value:", disabled=False
            )

            if shape == "(1,)":
                val = (
                    0.0
                    if default_wd is None
                    else default_wd["model"]["state_variables"]
                    .get(var_name, {})
                    .get("init_val_lower", 0.0)
                )
            else:
                val = (
                    0.0
                    if default_wd is None
                    else default_wd["model"]["state_variables"]
                    .get(var_name, {})
                    .get("init_val_lower", [0.0])[0]
                )
            sv_lower_bound = FloatText(
                value=val, description=f"Lower Bound:", disabled=False
            )
            if shape == "(1,)":
                val = (
                    0.0
                    if default_wd is None
                    else default_wd["model"]["state_variables"][var_name].get(
                        "init_val_upper", 0.0
                    )
                )
            else:
                val = (
                    0.0
                    if default_wd is None
                    else default_wd["model"]["state_variables"]
                    .get(var_name, {})
                    .get("init_val_upper", [0.0])[0]
                )
            sv_upper_bound = FloatText(
                value=val, description=f"Upper Bound:", disabled=False
            )

            sv_rhs = Text(
                value=(
                    "x1+x2"
                    if default_wd is None
                    else default_wd["model"]["state_variables"]
                    .get(var_name, {})
                    .get("rhs", "")
                ),
                placeholder="x1+x2",
                description="RHS:",
                disabled=False,
            )

            sv_shape = Text(
                value=shape, placeholder="(1,)", description="Shape:", disabled=False
            )

            sv_scaling = FloatText(
                value=(
                    1.0
                    if default_wd is None
                    else default_wd["model"]["state_variables"]
                    .get(var_name, {})
                    .get("scaling", 1.0)
                ),
                description=f"Scaling:",
                disabled=False,
            )

            sv_soft = Checkbox(
                value=(
                    False
                    if default_wd is None
                    else default_wd["model"]["state_variables"]
                    .get(var_name, {})
                    .get("soft", False)
                ),
                description="Soft",
                disabled=False,
                indent=False,
            )

            state_variables_children.extend(
                [
                    sv_def,
                    sv_init_val,
                    sv_lower_bound,
                    sv_upper_bound,
                    sv_rhs,
                    sv_shape,
                    sv_scaling,
                    sv_soft,
                ]
            )
            widget_dict.update(
                {
                    f"sv_{i}_def": sv_def,
                    f"sv_{i}_init_val": sv_init_val,
                    f"sv_{i}_lower_bound": sv_lower_bound,
                    f"sv_{i}_upper_bound": sv_upper_bound,
                    f"sv_{i}_rhs": sv_rhs,
                    f"sv_{i}_shape": sv_shape,
                    f"sv_{i}_scaling": sv_scaling,
                    f"sv_{i}_soft": sv_soft,
                }
            )
            state_variables_names.append(sv_def)
            state_variables_init_values.append(sv_init_val)
            state_variables_lower_bounds.append(sv_lower_bound)
            state_variables_upper_bounds.append(sv_upper_bound)
            state_variables_rhs.append(sv_rhs)
            state_variables_shape.append(sv_shape)
            state_variables_scaling.append(sv_scaling)
            state_variables_soft.append(sv_soft)
        grid = layout_generator(
            state_variables_header, state_variables_children, num_cols=8, col_width=200
        )
    return grid, widget_dict


def parameters_grid_generator(num_parameters, default_wd=None):
    widget_dict = {}
    grid = None
    if num_parameters > 0:
        parameters_children = []
        parameters_names = []
        parameters_uncertainties = []
        parameters_header = Button(
            description="User Defined Parameter Definition",
            layout=Layout(width="auto", grid_area="header"),
            style=ButtonStyle(button_color="lightblue"),
        )
        parameters_names = (
            [f"Parameter_{i}" for i in range(num_parameters)]
            if default_wd is None
            else list(default_wd["model"].get("user_defined_parameters", []))
        )
        for i in range(num_parameters):
            parameter_name = (
                parameters_names[i] if i < len(parameters_names) else f"Parameter_{i}"
            )
            para_def = Text(
                value=parameter_name,
                placeholder=parameter_name,
                description=f"{i+1}-th Parameter:",
                style={"description_width": "initial"},
                disabled=False,
            )
            para_val = str(
                "1,2,3"
                if default_wd is None
                else default_wd["mpc"]["uncertainties"].get(
                    parameters_names[i], "1,2,3"
                )
            )
            para_uncertainties = Text(
                value=para_val,
                placeholder=para_val,
                description="Values:",
                disabled=False,
            )

            parameters_children.extend([para_def, para_uncertainties])
            parameters_uncertainties.append(para_uncertainties)
            parameters_names.append(para_def)
            widget_dict.update(
                {
                    f"para_{i}_def": para_def,
                    f"para_{i}_uncertainties": para_uncertainties,
                }
            )
        grid = layout_generator(parameters_header, parameters_children, col_width=400)
        return grid, widget_dict


def control_variables_grid_generator(num_control_variables, default_wd=None):
    widget_dict = {}
    grid = None
    if num_control_variables > 0:
        control_variables_children = []
        control_variables_names = []
        control_variables_shapes = []
        control_variables_header = Button(
            description="Control Variables Definition",
            layout=Layout(width="auto", grid_area="header"),
            style=ButtonStyle(button_color="lightblue"),
        )
        control_variables = (
            [f"In_{i}" for i in range(num_control_variables)]
            if default_wd is None
            else list(default_wd["model"].get("control_variables", {}).keys())
        )
        print(num_control_variables)
        for i in range(num_control_variables):
            var_name = control_variables[i] if i < len(control_variables) else f"In_{i}"
            cv_def = Text(
                value=var_name,
                placeholder=var_name,
                description=f"{i+1}-th Control Variable:",
                style={"description_width": "initial"},
                disabled=False,
            )
            shape = (
                "(1,)"
                if default_wd is None
                else default_wd["model"]["control_variables"]
                .get(var_name, {})
                .get("shape", "(1,)")
            )
            cv_shape = Text(
                value=shape, placeholder=shape, description="Shape:", disabled=False
            )
            control_variables_children.extend([cv_def, cv_shape])
            control_variables_names.append(cv_def)
            control_variables_shapes.append(cv_shape)
            widget_dict.update({f"cv_{i}_def": cv_def, f"cv_{i}_shape": cv_shape})

        grid = layout_generator(
            control_variables_header, control_variables_children, col_width=400
        )
    return grid, widget_dict


def aux_variable_grid_generator(num_aux_variables, default_wd=None):
    widget_dict = {}
    grid = None
    if num_aux_variables > 0:
        aux_variables_children = []
        aux_variables_names = []
        aux_variables_expr = []
        aux_variables_header = Button(
            description="Auxiliary Variables Definition",
            layout=Layout(width="auto", grid_area="header"),
            style=ButtonStyle(button_color="lightblue"),
        )
        aux_variables_names = (
            [f"Aux_Variable_{i}" for i in range(num_aux_variables)]
            if default_wd is None
            else list(default_wd["model"].get("aux_variables", {}).keys())
        )
        for i in range(num_aux_variables):
            aux_name = (
                aux_variables_names[i]
                if i < len(aux_variables_names)
                else f"Aux_Variable_{i}"
            )
            av_def = Text(
                value=aux_name,
                placeholder=aux_name,
                description=f"{i+1}-th Aux. Variable:",
                style={"description_width": "initial"},
                disabled=False,
            )
            av_val = (
                f"Expr_{i}"
                if default_wd is None
                else default_wd["model"]["aux_variables"]
                .get(aux_name, {})
                .get("expr", aux_variables_names[i])
            )
            av_expr = Text(
                value=av_val, placeholder=av_val, description=f"Expr:", disabled=False
            )

            aux_variables_names.append(av_def)
            aux_variables_expr.append(av_expr)
            aux_variables_children.extend([av_def, av_expr])
            widget_dict.update({f"av_{i}_def": av_def, f"av_{i}_expr": av_expr})

        grid = layout_generator(
            aux_variables_header, aux_variables_children, col_width=400
        )
    return grid, widget_dict


def simulator_parameter_grid_generator(num_simulator_parameters, default_wd=None):
    widget_dict = {}
    grid = None
    if num_simulator_parameters > 0:
        num_simulator_parameters_children = []
        num_simulator_parameters_names = []
        num_simulator_parameters_values = []
        num_simulator_parameters_header = Button(
            description="Simulator Parameters",
            layout=Layout(width="auto", grid_area="header"),
            style=ButtonStyle(button_color="lightblue"),
        )
        parameter_names = (
            [f"Sim_Parameter_{i}" for i in range(num_simulator_parameters)]
            if default_wd is None
            else list(default_wd["simulator"].get("parameters", {}).keys())
        )
        for i in range(num_simulator_parameters):
            para_name = (
                parameter_names[i] if i < len(parameter_names) else f"Sim_Parameter_{i}"
            )
            sp_def = Text(
                value=para_name,
                placeholder=para_name,
                description=f"{i+1}-th Parameter of Simulator:",
                style={"description_width": "initial"},
                disabled=False,
            )
            val = (
                "val"
                if default_wd is None
                else default_wd["simulator"]["parameters"].get(
                    parameter_names[i], "val"
                )
            )
            sp_val = Text(
                value=str(val), placeholder=str(val), description="Value:", disabled=False
            )
            sp_numeric = Dropdown(
                options=["int", "float", "str", "bool"],
                value=type(val).__name__,
                description="Type:",
            )

            num_simulator_parameters_names.append(sp_def)
            num_simulator_parameters_values.append(sp_val)
            num_simulator_parameters_children.extend([sp_def, sp_val, sp_numeric])

            widget_dict.update(
                {
                    f"sim_para_{i}_def": sp_def,
                    f"sim_para_{i}_expr": sp_val,
                    f"sim_para_{i}_numeric": sp_numeric,
                }
            )
        grid = layout_generator(
            num_simulator_parameters_header,
            num_simulator_parameters_children,
            num_cols=3,
            col_width=400,
        )
    return grid, widget_dict


def state_rewards_grid_generator(sv_names, default_wd=None):
    widget_dict = {}
    grid = None
    if len(sv_names) > 0:
        state_reward_header = Button(
            description="Define State Reward",
            layout=Layout(width="auto", grid_area="header"),
            style=ButtonStyle(button_color="lightblue"),
        )
        step_rewards_expr = []
        step_rewards_coef = []
        terminal_rewards_expr = []
        terminal_rewards_coef = []
        state_rewards_children = []
        for i in range(len(sv_names)):
            sr_def = Label(value=sv_names[i], disabled=False)

            sr_expr = Text(
                value=(
                    f"({sv_names[i]}-0.6)"
                    if default_wd is None
                    else default_wd["reward"]["step_reward"]
                    .get(sv_names[i], {})
                    .get("expr", sv_names[i])
                ),
                placeholder=f"({sv_names[i]}-0.6)",
                description="Step Expr:",
                disabled=False,
            )

            sr_coef = FloatText(
                value=(
                    0.0
                    if default_wd is None
                    else default_wd["reward"]["step_reward"]
                    .get(sv_names[i], {})
                    .get("coef", 0.0)
                ),
                description=f"Step Coef:",
                disabled=False,
            )

            tr_expr = Text(
                value=(
                    f"({sv_names[i]}-0.6)"
                    if default_wd is None
                    else default_wd["reward"]["terminal_reward"]
                    .get(sv_names[i], {})
                    .get("expr", sv_names[i])
                ),
                placeholder=f"({sv_names[i]}-0.6)",
                description="Terminal Expr:",
                disabled=False,
            )

            tr_coef = FloatText(
                value=(
                    0.0
                    if default_wd is None
                    else default_wd["reward"]["terminal_reward"]
                    .get(sv_names[i], {})
                    .get("coef", 0.0)
                ),
                description=f"Terminal Coef:",
                style={"description_width": "initial"},
                disabled=False,
            )

            step_rewards_expr.append(sr_expr)
            step_rewards_coef.append(sr_coef)
            terminal_rewards_expr.append(tr_expr)
            terminal_rewards_coef.append(tr_coef)
            state_rewards_children.extend([sr_def, sr_expr, sr_coef, tr_expr, tr_coef])
            widget_dict.update(
                {
                    f"sr_{i}_def": sr_def,
                    f"sr_{i}_expr": sr_expr,
                    f"sr_{i}_coef": sr_coef,
                    f"tr_{i}_expr": tr_expr,
                    f"tr_{i}_coef": tr_coef,
                }
            )
        grid = layout_generator(
            state_reward_header, state_rewards_children, num_cols=5, col_width=250
        )
    return grid, widget_dict


def input_rewards_grid_generator(in_names, default_wd=None):
    widget_dict = {}
    grid = None
    if len(in_names) > 0:
        input_reward_header = Button(
            description="Define Input Reward",
            layout=Layout(width="auto", grid_area="header"),
            style=ButtonStyle(button_color="lightblue"),
        )
        input_rewards_coef = []
        input_rewards_children = []
        for i in range(len(in_names)):
            # in_def = Label(value=in_names[i], disabled=False)
            in_coef = FloatText(
                value=(
                    0.0
                    if default_wd is None
                    else default_wd["reward"]["input_reward"].get(in_names[i], 0.0)
                ),
                description=f"Coef of {in_names[i]}:",
                style={"description_width": "initial"},
                disabled=False,
            )
            input_rewards_coef.append(in_coef)
            input_rewards_children.extend([in_coef])
            widget_dict.update({f"in_{i}_coef": in_coef})

        grid = layout_generator(
            input_reward_header, input_rewards_children, num_cols=1, col_width=400
        )

    return grid, widget_dict


def MPC_config_grid_generator(sv_names, in_names, default_wd=None):
    widget_dict = {}
    grid = None
    if len(sv_names) + len(in_names) > 0:
        num_MPC_config_children = []
        num_MPC_config_header = Button(
            description="MPC Config",
            layout=Layout(width="auto", grid_area="header"),
            style=ButtonStyle(button_color="lightblue"),
        )
        for var_name in sv_names + in_names:
            mc_def = Text(
                value=var_name,
                placeholder=var_name,
                description=f"{var_name}:",
                style={"description_width": "initial"},
                disabled=False,
            )

            val = (
                0.0
                if default_wd is None
                else default_wd["mpc"]["bounds"].get(var_name, {}).get("lower", 0.0)
            )
            mc_lower_bound = FloatText(
                value=val, description=f"Lower Bound:", disabled=False
            )
            val = (
                0.0
                if default_wd is None
                else default_wd["mpc"]["bounds"].get(var_name, {}).get("upper", 10000.0)
            )
            mc_upper_bound = FloatText(
                value=val, description=f"Upper Bound:", disabled=False
            )

            mc_scaling = FloatText(
                value=(
                    1.0
                    if default_wd is None
                    else default_wd["mpc"]["bounds"]
                    .get(var_name, {})
                    .get("scaling", 1.0)
                ),
                description=f"Scaling:",
                disabled=False,
            )

            mc_soft = Checkbox(
                value=(
                    False
                    if default_wd is None
                    else default_wd["mpc"]["bounds"]
                    .get(var_name, {})
                    .get("soft", False)
                ),
                description="Soft",
                disabled=False,
                indent=False,
            )
            num_MPC_config_children.extend(
                [mc_def, mc_lower_bound, mc_upper_bound, mc_scaling, mc_soft]
            )
            widget_dict.update(
                {
                    f"mc_{var_name}_lower": mc_lower_bound,
                    f"mc_{var_name}_upper": mc_upper_bound,
                    f"mc_{var_name}_scaling": mc_scaling,
                    f"mc_{var_name}_soft": mc_soft,
                }
            )
        grid = layout_generator(
            num_MPC_config_header, num_MPC_config_children, num_cols=5, col_width=300
        )
    return grid, widget_dict


def MPC_parameter_grid_generator(num_MPC_parameters, default_wd=None):
    widget_dict = {}
    grid = None
    if num_MPC_parameters > 0:
        num_MPC_parameters_children = []
        num_MPC_parameters_names = []
        num_MPC_parameters_values = []
        num_MPC_parameters_header = Button(
            description="MPC Parameters",
            layout=Layout(width="auto", grid_area="header"),
            style=ButtonStyle(button_color="lightblue"),
        )
        parameter_names = (
            [f"MPC_Parameter_{i}" for i in range(num_MPC_parameters)]
            if default_wd is None
            else list(default_wd["mpc"].get("setup", {}).keys())
        )
        for i in range(num_MPC_parameters):
            para_name = (
                parameter_names[i] if i < len(parameter_names) else f"MPC_Parameter_{i}"
            )
            mp_def = Text(
                value=para_name,
                placeholder=para_name,
                description=f"{i+1}-th Parameter of MPC:",
                style={"description_width": "initial"},
                disabled=False,
            )
            val = (
                "val"
                if default_wd is None
                else default_wd["mpc"]["setup"].get(para_name, "val")
            )
            mp_val = Text(
                value=str(val), placeholder=str(val), description="Value:", disabled=False
            )

            mp_numeric = Dropdown(
                options=["int", "float", "str", "bool"],
                value=type(val).__name__,
                description="Type:",
            )
            num_MPC_parameters_names.append(mp_def)
            num_MPC_parameters_values.append(mp_val)
            num_MPC_parameters_children.extend([mp_def, mp_val, mp_numeric])

            widget_dict.update(
                {
                    f"MPC_para_{i}_def": mp_def,
                    f"MPC_para_{i}_expr": mp_val,
                    f"MPC_para_{i}_numeric": mp_numeric,
                }
            )
        grid = layout_generator(
            num_MPC_parameters_header,
            num_MPC_parameters_children,
            num_cols=3,
            col_width=400,
        )
    return grid, widget_dict


def estimator_parameter_grid_generator(num_estimator_parameters, default_wd=None):
    widget_dict = {}
    grid = None
    num_estimator_parameters_children = []
    num_estimator_parameters_names = []
    num_estimator_parameters_values = []
    num_estimator_parameters_header = Button(
        description="Estimator Parameters",
        layout=Layout(width="auto", grid_area="header"),
        style=ButtonStyle(button_color="lightblue"),
    )

    est_type = Dropdown(
        options=["StateFeedback", "EKF", "MHE"],
        value=(
            "StateFeedback" if default_wd is None else default_wd["estimator"]["type"]
        ),
        description="Estimator:",
        disabled=False,
    )
    widget_dict["est_type"] = est_type
    label_description = Label(value="Set the type of estimator.")
    label_choice = Label(value="Choose from ['StateFeedback', 'EKF', 'MHE']")
    num_estimator_parameters_children.extend(
        [label_description, label_choice, est_type]
    )
    if default_wd is None:
        parameter_names = [
            f"Estimator_Parameter_{i}" for i in range(num_estimator_parameters)
        ]
        parameters_dd = {}
    else:
        parameters_dd = default_wd["estimator"].get("parameters", {})
        if parameters_dd is None:
            parameters_dd = {}
        parameter_names = list(parameters_dd.keys())

    for i in range(num_estimator_parameters):
        para_name = (
            parameter_names[i]
            if i < len(parameter_names)
            else f"Estimator_Parameter_{i}"
        )
        ep_def = Text(
            value=para_name,
            placeholder=para_name,
            description=f"{i+1}-th Parameter of estimator:",
            style={"description_width": "initial"},
            disabled=False,
        )

        val = parameters_dd.get(para_name, "val")
        ep_val = Text(value=str(val), placeholder=str(val), description="Value:", disabled=False)
        ep_numeric = Dropdown(
            options=["int", "float", "str", "bool"],
            value=type(val).__name__,
            description="Type:",
        )

        num_estimator_parameters_names.append(ep_def)
        num_estimator_parameters_values.append(ep_val)
        num_estimator_parameters_children.extend([ep_def, ep_val, ep_numeric])

        widget_dict.update(
            {
                f"estimator_para_{i}_def": ep_def,
                f"estimator_para_{i}_expr": ep_val,
                f"estimator_para_{i}_numeric": ep_numeric,
            }
        )
    grid = layout_generator(
        num_estimator_parameters_header,
        num_estimator_parameters_children,
        num_cols=3,
        col_width=400,
    )
    return grid, widget_dict


def model2yaml(wd, model_config):
    for i in range(wd["num_constants"].value):
        model_config["model"]["constants"][wd[f"constant_{i}_def"].value] = wd[
            f"constant_{i}_val"
        ].value

    for i in range(wd["num_state_variables"].value):
        shape = wd[f"sv_{i}_shape"].value
        model_config["model"]["state_variables"][wd[f"sv_{i}_def"].value] = {
            "init_val": (
                wd[f"sv_{i}_init_val"].value
                if shape == "(1,)"
                else [wd[f"sv_{i}_init_val"].value]
            ),
            "init_val_lower": (
                wd[f"sv_{i}_lower_bound"].value
                if shape == "(1,)"
                else [wd[f"sv_{i}_lower_bound"].value]
            ),
            "init_val_upper": (
                wd[f"sv_{i}_upper_bound"].value
                if shape == "(1,)"
                else [wd[f"sv_{i}_upper_bound"].value]
            ),
            "rhs": wd[f"sv_{i}_rhs"].value,
            "shape": wd[f"sv_{i}_shape"].value,
            "scaling": wd[f"sv_{i}_scaling"].value,
        }

    for i in range(wd["num_control_variables"].value):
        model_config["model"]["control_variables"][wd[f"cv_{i}_def"].value] = {
            "shape": wd[f"cv_{i}_shape"].value
        }

    model_config["model"]["user_defined_parameters"] = [
        wd[f"para_{i}_def"].value for i in range(wd["num_parameters"].value)
    ]

    for i in range(wd["num_aux_variables"].value):
        model_config["model"]["aux_variables"][wd[f"av_{i}_def"].value] = {
            "expr": wd[f"av_{i}_expr"].value
        }
    return model_config


def simulator2yaml(wd, model_config):
    model_config["simulator"] = {"parameters": {}}
    for i in range(wd["num_simulator_parameters"]):
        type_eval = eval(wd[f"sim_para_{i}_numeric"].value)
        if type_eval is bool:
            model_config["simulator"]["parameters"][
                wd[f"sim_para_{i}_def"].value
            ] = eval_bool(wd[f"sim_para_{i}_expr"].value)
        else:
            model_config["simulator"]["parameters"][
                wd[f"sim_para_{i}_def"].value
            ] = type_eval(wd[f"sim_para_{i}_expr"].value)
    return model_config


def reward2yaml(wd, model_config):
    model_config["reward"] = {
        "step_reward": {},
        "terminal_reward": {},
        "input_reward": {
            wd[f"cv_{i}_def"].value: wd[f"in_{i}_coef"].value
            for i in range(wd["num_control_variables"].value)
        },
    }
    for i in range(wd["num_state_variables"].value):
        model_config["reward"]["step_reward"][wd[f"sv_{i}_def"].value] = {
            "expr": wd[f"sr_{i}_expr"].value,
            "coef": wd[f"sr_{i}_coef"].value,
        }
        model_config["reward"]["terminal_reward"][wd[f"sv_{i}_def"].value] = {
            "expr": wd[f"tr_{i}_expr"].value,
            "coef": wd[f"tr_{i}_coef"].value,
        }
    return model_config


def mpc2yaml(wd, model_config):
    model_config["mpc"] = {"setup": {}}
    for i in range(wd["num_MPC_parameters"]):
        type_eval = eval(wd[f"MPC_para_{i}_numeric"].value)
        if type_eval is bool:
            model_config["mpc"]["setup"][wd[f"MPC_para_{i}_def"].value] = eval_bool(
                wd[f"MPC_para_{i}_expr"].value
            )
        else:
            model_config["mpc"]["setup"][wd[f"MPC_para_{i}_def"].value] = type_eval(
                wd[f"MPC_para_{i}_expr"].value
            )

    sv_names = [wd[f"sv_{i}_def"].value for i in range(wd["num_state_variables"].value)]
    in_names = [
        wd[f"cv_{i}_def"].value for i in range(wd["num_control_variables"].value)
    ]
    model_config["mpc"]["scaling"] = {}
    model_config["mpc"]["bounds"] = {}
    model_config["mpc"]["uncertainties"] = {}
    for var_name in sv_names + in_names:
        val_key = f"mc_{var_name}_scaling"
        if val_key in wd:
            model_config["mpc"]["scaling"][var_name] = wd[val_key].value
        else:
            model_config["mpc"]["scaling"][var_name] = 1.0
        model_config["mpc"]["bounds"][var_name] = {}
        for default_val, key in [(0.0, "lower"), (10000.0, "upper"), (False, "soft")]:
            val_key = f"mc_{var_name}_{key}"
            if val_key in wd:
                model_config["mpc"]["bounds"][var_name][key] = wd[val_key].value
            else:
                model_config["mpc"][key][var_name][key] = default_val

    num_parameters = wd["num_parameters"].value
    for i in range(num_parameters):
        para_name = wd[f"para_{i}_def"].value
        para_val = eval(wd[f"para_{i}_uncertainties"].value)
        model_config["mpc"]["uncertainties"][para_name] = para_val
    return model_config


def estimator2yaml(wd, model_config):
    model_config["estimator"] = {"type": wd["est_type"].value, "parameters": {}}
    for i in range(wd["num_estimator_parameters"]):
        type_eval = eval(wd[f"estimator_para_{i}_numeric"].value)
        if type_eval is bool:
            model_config["estimator"]["parameters"][
                wd[f"estimator_para_{i}_def"].value
            ] = eval_bool(wd[f"estimator_para_{i}_expr"].value)
        else:
            model_config["estimator"]["parameters"][
                wd[f"estimator_para_{i}_def"].value
            ] = type_eval(wd[f"estimator_para_{i}_expr"].value)
    return model_config


def save2yaml(wd, yaml_file):
    model_config = {}
    model_config["model"] = {
        "model_name": wd["model_name"].value,
        "model_type": wd["model_type"].value,
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

    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(model_config, f)
