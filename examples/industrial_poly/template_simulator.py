#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_simulator(model):
    """
    --------------------------------------------------------------------------
    template_simulator: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 50.0/3600.0
    }

    simulator.set_param(**params_simulator)

    p_num = simulator.get_p_template()
    p_num['delH_R'] = 950
    p_num['k_0'] = 7
    delH_R_var = np.array([950.0, 950.0 * 1.30, 950.0 * 0.70])
    k_0_var = np.array([7.0*1.00, 7.0*1.30, 7.0*0.70])
    def p_fun(t_now):
        p_num['delH_R'] = np.random.choice(delH_R_var)
        p_num['k_0'] = np.random.choice(k_0_var)
        return p_num
    
    simulator.set_p_fun(p_fun)
    simulator.setup()

    return simulator


def reward_function(cur_step, simulator_data):
    reward = simulator_data["_x", "m_P"][-1, 0]
    reward -= 1e4*max(0, simulator_data['_x', 'T_R'][-1, 0]-365.15)
    if cur_step > 0:
        reward -= 0.002*(simulator_data['_u', 'm_dot_f'][-2, 0] - simulator_data['_u', 'm_dot_f'][-1, 0])**2
        reward -= 0.004*(simulator_data['_u', 'T_in_M'][-2, 0] - simulator_data['_u', 'T_in_M'][-1, 0])**2
        reward -= 0.002*(simulator_data['_u', 'T_in_EK'][-2, 0] - simulator_data['_u', 'T_in_EK'][-1, 0])**2
    return reward