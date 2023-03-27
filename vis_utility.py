from ipywidgets import *


def layout_generator(header, children, num_cols=2, col_width=200, row_height=40):
    num_rows = len(children) // num_cols
    aux_cols = 1
    if len(children) % num_cols:
        footer = children[-1]
        aux_cols += 1
    else: footer = None
    
    grid = GridspecLayout(num_rows+aux_cols, num_cols, width=f"{col_width*num_cols}px", height=f'{row_height*(num_rows+aux_cols)}px', merge=True)
    grid[0, :] = header
    for row in range(num_rows):
         for col in range(num_cols):
            grid[row+1, col] = children[row*num_cols+col]
    if footer: 
        for col in range(num_cols):
            if num_rows*num_cols + col < len(children): grid[-1, col] = children[num_rows*num_cols + col]
    return grid



def var_num_grid_generator():
    header  = Button(description='Model Specification',
                 layout=Layout(width='auto', grid_area='header'),
                 style=ButtonStyle(button_color='lightblue'))
    model_name = Text(value='Model Name',
                    placeholder='Type something',
                    description='Model Name:',
                    disabled=False)

    model_type = Dropdown(options=['continuous', 'discrete'],
                        value='continuous',
                        description='Model Type:',
                        disabled=False)


    num_constants = BoundedIntText(value=2,
                                min=0,
                                step=1,
                                description='Num. Constants:',
                                disabled=False)


    num_state_variables = BoundedIntText(value=2,
                                        min=0,
                                        step=1,
                                        description='Num. State Variables:',
                                        disabled=False)



    num_control_variables = BoundedIntText(value=2,
                                        min=0,
                                        step=1,
                                        description='Num. Control Variables:',
                                        disabled=False)


    num_parameters = BoundedIntText(value=2,
                                    min=0,
                                    step=1,
                                    description='Num. User Defined Parameters',
                                    disabled=False)


    num_aux_variables = BoundedIntText(value=2,
                                    min=0,
                                    step=1,
                                    description='Num. Auxiliary Variables:',
                                    disabled=False)

    widget_dict = {"model_name": model_name, 
                   "model_type": model_type, 
                   "num_constants": num_constants,
                   "num_state_variables": num_state_variables, 
                   "num_control_variables": num_control_variables,
                   "num_parameters": num_parameters, 
                   "num_aux_variables": num_aux_variables}

    grid = layout_generator(header, list(widget_dict.values()), col_width=400)
    return grid, widget_dict


def constants_grid_generator(num_constants):
    widget_dict = {}
    grid = None
    if num_constants > 0:
        constant_children = []
        constant_names = []
        constant_values = []
        constants_header = Button(description='Constants Definition',
                        layout=Layout(width='auto', grid_area='header'),
                        style=ButtonStyle(button_color='lightblue'))
        for i in range(num_constants):
            constant_def = Text(value=f"X_{i}",
                                placeholder=f"X_{i}",
                                description=f"{i+1}-th Constant Name:",
                                disabled=False)

            constant_val = FloatText(value=0.0,
                                    description=f"Value:",
                                    disabled=False)
            constant_children.extend([constant_def, constant_val])
            widget_dict.update({f"constant_{i}_def": constant_def,
                                f"constant_{i}_val": constant_val})
            constant_names.append(constant_def)
            constant_values.append(constant_val)


        grid = layout_generator(constants_header, constant_children, col_width=400)
    return grid, widget_dict
    
    
def state_variables_grid_generator(num_state_variables):
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
        
        state_variables_header = Button(description='State Variables Definition',
                        layout=Layout(width='auto', grid_area='header'),
                        style=ButtonStyle(button_color='lightblue'))
        for i in range(num_state_variables):
            sv_def = Text(value=f"X_{i}",
                                placeholder=f"X_{i}",
                                description=f"{i+1}-th State Variable Name:",
                                disabled=False)

            sv_init_val = FloatText(value=0.0,
                                    description=f"Init Value:",
                                    disabled=False)
            
            sv_lower_bound = FloatText(value=0.0,
                                    description=f"Lower Bound:",
                                    disabled=False)
            
            sv_upper_bound = FloatText(value=0.0,
                                    description=f"Upper Bound:",
                                    disabled=False)
            
            sv_rhs = Text(value="x1+x2",
                                placeholder="x1+x2",
                                description="RHS:",
                                disabled=False)

            sv_shape = Text(value="(1,)",
                            placeholder="(1,)",
                            description="Shape:",
                            disabled=False)
                        
            sv_scaling = FloatText(value=1.0,
                                description=f"Scaling:",
                                disabled=False)
            
            sv_soft = Checkbox(value=False,
                            description='Soft',
                            disabled=False,
                            indent=False)

            
            state_variables_children.extend([sv_def, sv_init_val, sv_lower_bound, sv_upper_bound, sv_rhs, sv_shape, sv_scaling, sv_soft])
            widget_dict.update({
                f"sv_{i}_def": sv_def, 
                f"sv_{i}_init_val": sv_init_val, 
                f"sv_{i}_lower_bound": sv_lower_bound, 
                f"sv_{i}_upper_bound": sv_upper_bound, 
                f"sv_{i}_rhs": sv_rhs, 
                f"sv_{i}_shape": sv_shape, 
                f"sv_{i}_scaling": sv_scaling, 
                f"sv_{i}_soft": sv_soft
            })
            state_variables_names.append(sv_def)
            state_variables_init_values.append(sv_init_val)
            state_variables_lower_bounds.append(sv_lower_bound)
            state_variables_upper_bounds.append(sv_upper_bound)
            state_variables_rhs.append(sv_rhs)
            state_variables_shape.append(sv_shape)
            state_variables_scaling.append(sv_scaling)
            state_variables_soft.append(sv_soft)
        grid = layout_generator(state_variables_header, state_variables_children, num_cols=8, col_width=200)
    return grid, widget_dict
    
    
def parameters_grid_generator(num_parameters):
    widget_dict = {}
    grid = None
    if num_parameters > 0:
        parameters_children = []
        parameters_names = []
        parameters_uncertainities = []
        parameters_header = Button(description='User Defined Parameter Definition',
                        layout=Layout(width='auto', grid_area='header'),
                        style=ButtonStyle(button_color='lightblue'))
        for i in range(num_parameters):
            para_def = Text(value=f"Parameter_{i}",
                        placeholder=f"Parameter_{i}",
                        description=f"{i+1}-th Parameter:",
                        disabled=False)
            
            para_uncertainties = Text(value="1,2,3",
                                    placeholder="1,2,3",
                                    description="Values:",
                                    disabled=False)

            parameters_children.extend([para_def, para_uncertainties])
            parameters_uncertainities.append(para_uncertainties)
            parameters_names.append(para_def)
            widget_dict.update({f"para_{i}_def": para_def,
                                f"para_{i}_uncertainities": para_uncertainties})
        grid = layout_generator(parameters_header, parameters_children, col_width=400)
        return grid, widget_dict
    
    
def control_variables_grid_generator(num_control_variables):
    widget_dict = {}
    grid = None
    if num_control_variables > 0:
        control_variables_children = []
        control_variables_names = []
        control_variables_shapes = []
        control_variables_header = Button(description='Control Variables Definition',
                        layout=Layout(width='auto', grid_area='header'),
                        style=ButtonStyle(button_color='lightblue'))
        for i in range(num_control_variables):
            cv_def = Text(value=f"In_{i}",
                        placeholder=f"In_{i}",
                        description=f"{i+1}-th Control Variable:",
                        disabled=False)

            cv_shape = Text(value="(1,)",
                            placeholder="(1,)",
                            description="Shape:",
                            disabled=False)
            control_variables_children.extend([cv_def, cv_shape])
            control_variables_names.append(cv_def)
            control_variables_shapes.append(cv_shape)
            widget_dict.update({f"cv_{i}_def": cv_def,
                                f"cv_{i}_shape": cv_shape})

        grid = layout_generator(control_variables_header, control_variables_children, col_width=400)
    return grid, widget_dict


def aux_variable_grid_generator(num_aux_variables):
    widget_dict = {}
    grid = None
    if num_aux_variables > 0:
        aux_variables_children = []
        aux_variables_names = []
        aux_variables_expr = []
        aux_variables_header = Button(description='Auxiliary Variables Definition',
                        layout=Layout(width='auto', grid_area='header'),
                        style=ButtonStyle(button_color='lightblue'))
        for i in range(num_aux_variables):
            av_def = Text(value=f"Aux_Variable_{i}",
                        placeholder=f"Aux_Variable_{i}",
                        description=f"{i+1}-th Aux. Variable:",
                        disabled=False)
            
            av_expr = Text(value=f"Expr_{i}",
                        placeholder=f"Expr_{i}",
                        description=f"Expr:",
                        disabled=False)

            aux_variables_names.append(av_def)
            aux_variables_expr.append(av_expr)
            aux_variables_children.extend([av_def, av_expr])
            widget_dict.update({f"av_{i}_def": av_def,
                                f"av_{i}_expr": av_expr})

        grid = layout_generator(aux_variables_header, aux_variables_children, col_width=400)
    return grid, widget_dict


def simulator_parameter_grid_generator(num_simulator_parameters):
    widget_dict = {}
    grid = None
    if num_simulator_parameters > 0:
        num_simulator_parameters_children = []
        num_simulator_parameters_names = []
        num_simulator_parameters_values = []
        num_simulator_parameters_header = Button(description='Simulator Parameters',
                        layout=Layout(width='auto', grid_area='header'),
                        style=ButtonStyle(button_color='lightblue'))
        for i in range(num_simulator_parameters):
            sp_def = Text(value=f"Sim_Parameter_{i}",
                        placeholder=f"Sim_Parameter_{i}",
                        description=f"{i+1}-th Parameter of Simulator:",
                        disabled=False)
            
            sp_val = Text(value="val",
                        placeholder="val",
                        description="Value:",
                        disabled=False)

            sp_numeric =  Checkbox(value=False,
                            description='Numeric',
                            disabled=False,
                            indent=False)
            
            num_simulator_parameters_names.append(sp_def)
            num_simulator_parameters_values.append(sp_val)
            num_simulator_parameters_children.extend([sp_def, sp_val, sp_numeric])

            widget_dict.update({f"sim_para_{i}_def": sp_def,
                                f"sim_para_{i}_expr": sp_val,
                                f"sim_para_{i}_numeric": sp_numeric})
        grid = layout_generator(num_simulator_parameters_header, num_simulator_parameters_children, num_cols=3, col_width=400)
    return grid, widget_dict


def state_rewards_grid_generator(sv_names):
    widget_dict = {}
    grid = None
    if len(sv_names) > 0:
        state_reward_header = Button(description='Define State Reward',
                                layout=Layout(width='auto', grid_area='header'),
                                style=ButtonStyle(button_color='lightblue'))
        step_rewards_expr = []
        step_rewards_coef = []
        terminal_rewards_expr = []
        terminal_rewards_coef = []
        state_rewards_children = []
        for i in range(len(sv_names)):
            sr_def = Label(value=sv_names[i],
                        disabled=False)

            sr_expr = Text(value=f"({sv_names[i]}-0.6)",
                                placeholder=f"({sv_names[i]}-0.6)",
                                description="Step Expr:",
                                disabled=False)

            sr_coef = FloatText(value=0.0,
                                description=f"Step Coef:",
                                disabled=False)
            
            tr_expr = Text(value=f"({sv_names[i]}-0.6)",
                                placeholder=f"({sv_names[i]}-0.6)",
                                description="Terminal Expr:",
                                disabled=False)

            tr_coef = FloatText(value=0.0,
                                description=f"Terminal Coef:",
                                disabled=False)
            
            step_rewards_expr.append(sr_expr)
            step_rewards_coef.append(sr_coef)
            terminal_rewards_expr.append(tr_expr)
            terminal_rewards_coef.append(tr_coef)
            state_rewards_children.extend([sr_def, sr_expr, sr_coef, tr_expr, tr_coef])
            widget_dict.update({
                f"sr_{i}_def": sr_def,  
                f"sr_{i}_expr": sr_expr, 
                f"sr_{i}_coef": sr_coef, 
                f"tr_{i}_expr": tr_expr, 
                f"tr_{i}_coef": tr_coef
            })
        grid = layout_generator(state_reward_header, state_rewards_children, num_cols=5, col_width=250)
    return grid, widget_dict



def input_rewards_grid_generator(in_names):
    widget_dict = {}
    grid = None
    if len(in_names) > 0:
        input_reward_header = Button(description='Define Input Reward',
                                layout=Layout(width='auto', grid_area='header'),
                                style=ButtonStyle(button_color='lightblue'))
        input_rewards_coef = []
        input_rewards_children = []
        for i in range(len(in_names)):
            in_def = Label(value=in_names[i], disabled=False)

            in_coef = FloatText(value=0.0,
                                description=f"Coef:",
                                disabled=False)
            input_rewards_coef.append(in_coef)
            input_rewards_children.extend([in_def, in_coef])
            widget_dict.update({f"in_{i}_def": in_def,
                                f"in_{i}_coef": in_coef})

        grid = layout_generator(input_reward_header, input_rewards_children, num_cols=2, col_width=400)
    
    return grid, widget_dict