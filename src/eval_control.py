#
#
#
#
#
#
#
#
from flow_solver import FlowSolver
import numpy as np


def eval_control(control, params):
    '''Evolve environment by f: (probe readings) -> scalar'''
    # Setup flow solver
    solver = FlowSolver(params['flow'], params['geometry'], params['solver'])

    # Setup probes
    nprobes = len(params['optim']['probe_positions'])
    pressures = PressureProbeANN(solver, params['optim']['probe_positions'])

    # Keep track of one jet value. Other is -jet
    Q = 0
    uh, ph, status = solver.evolve(Q*np.ones(nprobes))
    
    # Evolve the environment
    T_terminal = params['optim']['T_term']
    alpha = params['optim']['smooth_control']
    while solver.gtime < T_terminal:

        pressure_values = pressures.sample(uh, ph)
        # Finite?

        action = control(*pressure_values)
        for _ in range(nsmooth_steps):
            Q += alpha*(action - Q)
            uh, ph, status = solver.evolve(np.array([Q, -Q]))
            # Finite?

            # Instant drag

    # Last

# eval without deap expr
#  - test
#
# eval for deap
#  - test
#
# add deap

# ---------------------------------------------------------------------

if __name__ == '__main__':

    # Flow solver parameters
    geometry_params = {'jet_radius': 0.05,
                       'jet_positions': [90, 270],
                       'jet_width': 10,
                       'mesh': './mesh/mesh.h5'}

    solver_params = {'dt': dt}

    flow_params = {'mu': 1E-3,
                   'rho': 1,
                   'u0_file': './mesh/u_init.xdmf',
                   'p0_file': './mesh/p_init.xdmf'}

    # Optimization
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
    # Kick the front and back
    angles = angles[[2, 3, 7, 8]]
    probe_positions = np.c_[radius*np.cos(angles), radius*np.sin(angles)]

    optim_params = {'T_term': 2,
                    'ncontrol': 80,
                    'smooth_control': x,
                    'probe_positions': probe_positions}

                            "smooth_control": (nb_actuations/dt)*(0.1*0.0005/80),
