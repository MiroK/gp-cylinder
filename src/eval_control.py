from probes import PressureProbeANN, DragProbeANN, LiftProbeANN
from flow_solver import FlowSolver
from dolfin import File
import numpy as np


class VTKIO(object):
    '''Dump to VTK at intervals'''
    def __init__(self, root, fields, period):
        self.out = [File('%s_%s.pvd' % (root, f)) for f in fields]
        self.counter = 0
        self.period = period

    def __call__(self, fields):
        if self.counter % self.period == 0:
            for out, field in zip(self.out, fields):
                out << (field, float(self.counter))
        self.counter += 1

        
class NumpyIO(object):
    '''Dump dict of histories as numpy array at intervals'''
    def __init__(self, path, period):
        self.counter = 0
        self.last_slice = 0
        self.path = path
        self.period = period
        self.keys = None

    def __call__(self, dct):
        if self.keys is None: self.keys = dct.keys()

        if self.counter % self.period == 0:
            # Data of equal len
            data = []
            l = NumpyIO.data_len(dct)
            for key in self.keys:
                data.append(np.array(dct[key][self.last_slice:l]))
            data = np.column_stack(data)

            if self.counter == 0:
                np.savetxt(self.path, data, header=' '.join(map(str, self.keys)))
            else:
                with open(self.path, 'a') as f: np.savetxt(f, data)

            self.last_slice = l
        self.counter += 1

    @staticmethod
    def data_len(dct):
        return min(map(len, dct.values()))

    
def isfinite(values):
    '''Sane number'''
    return not np.any(np.isinf(values)) and not np.any(np.isnan(values))


def eval_control(control, params, history=None, vtk_io=None, np_io=None):
    '''Evolve environment by f: (t, probe readings) -> scalar'''
    #
    # Setup flow solver
    solver = FlowSolver(params['flow'], params['geometry'], params['solver'])

    # Setup probes
    pressures = PressureProbeANN(solver, params['optim']['probe_positions'])
    drag = DragProbeANN(solver)
    lift = LiftProbeANN(solver)

    # Keep track of one jet value. Other is -jet
    Q = 0
    uh, ph, status = solver.evolve(Q*np.ones(len(params['geometry']['jet_positions'])))
    
    # Evolve the environment
    T_terminal = params['optim']['T_term']
    alpha = params['optim']['smooth_control']
    nsmooth_steps = int(params['optim']['dt_control']/params['solver']['dt'])
    assert nsmooth_steps >= 1

    # Record history of control, readings, drag ...
    time_past, control_past, pressure_past, drag_past, lift_past = [], [], [], [], []
    # All
    past = {'time': time_past,
            'control': control_past,
            'pressure': pressure_past,
            'drag': drag_past,
            'lift': lift_past}
    
    last_state = lambda past=past: tuple(past[key][-1] for key in past)
    
    evolve_okay = True
    while solver.gtime < T_terminal and evolve_okay:

        pressure_values = pressures.sample(uh, ph)
        
        evolve_okay = isfinite(pressure_values)
        if not evolve_okay: break
        
        action = control(solver.gtime, *pressure_values)
        for _ in range(nsmooth_steps):
            # Smoothed action (like RL)
            if alpha:
                Q += alpha*(action - Q)
            else:
                Q = action
            # At time t the action was Q and it resulted in pressure and drag
            time_past.append(solver.gtime)
            control_past.append(Q)
                
            uh, ph, status = solver.evolve(np.array([Q, -Q]))
            pressure_past.append(np.hstack(pressures.sample(uh, ph)).tolist())
            # Debug
            vtk_io is not None and vtk_io((uh, ph))
            
            # Instant 
            drag_value, lift_value = drag.sample(uh, ph), lift.sample(uh, ph)
            drag_past.append(drag_value)
            lift_past.append(lift_value)
            # Debug
            np_io is not None and np_io(past)

            # Finite?
            evolve_okay = all((status, isfinite(drag_value), isfinite(lift_value)))
            if not evolve_okay: break
            
        if not evolve_okay: break

        print last_state()
        
    # Accumulated drag (like RL)
    avg_length = min(500, len(drag_past))
    avg_abs_lift = np.mean(lift_past[-avg_length:])
    avg_drag = np.mean(drag_past[-avg_length:])
    
    score =  -avg_drag # + 0.159 - 0.2*avg_abs_lift

    # Weight the cost by when things broke
    if not evolve_okay:
        score *= np.exp(T_terminal/solver.gtime)/np.exp(1)

    # Fix history
    if history is not None:
        l = min(*map(len, past.values()))
    
        for field, f in past.items():
            history[field] = f[:l]

    return (score, evolve_okay)


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

    solver_params = {'dt': 5E-4}

    flow_params = {'mu': 1E-3,
                   'rho': 1,
                   'u0_file': './mesh/u_init.xdmf',
                   'p0_file': './mesh/p_init.xdmf'}

    # Optimization
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
    # Kick the front and back
    angles = angles[[2, 3, 7, 8]]
    radius = geometry_params['jet_radius']
    probe_positions = np.c_[radius*np.cos(angles), radius*np.sin(angles)]

    optimization_params = {'T_term': 2,
                           'dt_control': 20*solver_params['dt'],
                           'smooth_control': 0.1,
                           'probe_positions': probe_positions}

    params = {'geometry': geometry_params,
              'flow': flow_params,
              'solver': solver_params,
              'optim': optimization_params}

    history = {}
    control = lambda t, p0, p1, p2, p3: 1E-2*np.sin(4*np.pi*t)

    vtk_io = VTKIO('./results/test/test_', ['uh', 'ph'], 1)
    np_io = NumpyIO('./results/test/test.txt', 20)
    
    print eval_control(control, params, history, vtk_io=vtk_io, np_io=np_io)
