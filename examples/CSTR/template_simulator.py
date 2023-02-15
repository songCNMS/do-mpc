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
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 0.005
    }

    simulator.set_param(**params_simulator)

    tvp_num = simulator.get_tvp_template()
    def tvp_fun(t_now):
        return tvp_num

    simulator.set_tvp_fun(tvp_fun)

    alpha_var = np.array([1., 1.05, 0.95])
    beta_var = np.array([1., 1.1, 0.9])
    
    p_num = simulator.get_p_template()
    p_num['alpha'] = 1
    p_num['beta'] = 1
    def p_fun(t_now):
        p_num['alpha'] = np.random.choice(alpha_var)
        p_num['beta'] = np.random.choice(beta_var)
        return p_num

    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator


def reward_function(cur_step, simulator_data):
    reward = -(simulator_data['_x', 'C_b'][-1, 0] - 0.6)**2
    reward -= 1e2*max(0, simulator_data['_x', 'T_R'][-1, 0]-140)
    if cur_step > 0:
        reward -= 0.1*(simulator_data['_u', 'F'][-2, 0] - simulator_data['_u', 'F'][-1, 0])**2
        reward -= 1e-3*(simulator_data['_u', 'Q_dot'][-2, 0] - simulator_data['_u', 'Q_dot'][-1, 0])**2
    return reward