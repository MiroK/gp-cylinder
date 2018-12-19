from dolfin import *
import numpy as np


class FlowSolver(object):
    '''IPCS scheme with explicit treatment of nonlinearity.'''
    def __init__(self, flow_params, geometry_params, solver_params):
        # Using very simple IPCS solver
        mu = Constant(flow_params['mu'])              # dynamic viscosity
        rho = Constant(flow_params['rho'])            # density

        mesh_file = geometry_params['mesh']
        print(mesh_file)

        # Load mesh with markers
        comm = mpi_comm_world()
        h5 = HDF5File(comm, mesh_file, 'r')
        mesh = Mesh()
        h5.read(mesh, 'mesh', False)

        surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
        h5.read(surfaces, 'facet')

        # These tags should be hardcoded by gmsh during generation
        inlet_tag = 3
        outlet_tag = 2
        wall_tag = 1
        cylinder_noslip_tag = 4  # The entire surface

        # Define function spaces
        V = VectorFunctionSpace(mesh, 'CG', 2)
        Q = FunctionSpace(mesh, 'CG', 1)

        # Define trial and test functions
        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)

        u_n, p_n = Function(V), Function(Q)
        # Starting from rest or are we given the initial state
        for path, func, name in zip(('u_init', 'p_init'), (u_n, p_n), ('u0', 'p0')):
            if path in flow_params:
                comm = mesh.mpi_comm()

                XDMFFile(comm, flow_params[path]).read_checkpoint(func, name, 0)
                # assert func.vector().norm('l2') > 0

        u_, p_ = Function(V), Function(Q)  # Solve into these

        dt = Constant(solver_params['dt'])
        # Define expressions used in variational forms
        U  = Constant(0.5)*(u_n + u)
        n  = FacetNormal(mesh)
        f  = Constant((0, 0))

        epsilon = lambda u :sym(nabla_grad(u))

        sigma = lambda u, p: 2*mu*epsilon(u) - p*Identity(2)

        # Define variational problem for step 1
        F1 = (rho*dot((u - u_n) / dt, v)*dx
              + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx
              + inner(sigma(U, p_n), epsilon(v))*dx
              + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds
              - dot(f, v)*dx)

        a1, L1 = lhs(F1), rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/dt)*div(u_)*q*dx

        # Define variational problem for step 3
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - dt*dot(nabla_grad(p_ - p_n), v)*dx

        inflow_profile = flow_params['inflow_profile'](mesh, degree=2)
        # Define boundary conditions, first those that are constant in time
        bcu_inlet = DirichletBC(V, inflow_profile, surfaces, inlet_tag)
        # No slip
        bcu_wall = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag)
        # Fixing outflow pressure
        bcp_outflow = DirichletBC(Q, Constant(0), surfaces, outlet_tag)
        # Rotation of the cylinder
        cylinder_bc_value = Expression(('A*x[1]/sqrt(x[0]*x[0] + x[1]*x[1])',
                                        '-A*x[0]/sqrt(x[0]*x[0] + x[1]*x[1])'), degree=1, A=0)
        bcu_cylinder = [DirichletBC(V, cylinder_bc_value, surfaces, cylinder_noslip_tag)]

        # All bcs objects togets
        bcu = [bcu_inlet, bcu_wall] + bcu_cylinder
        bcp = [bcp_outflow]

        As = [Matrix() for i in range(3)]
        bs = [Vector() for i in range(3)]

        # Assemble matrices
        assemblers = [SystemAssembler(a1, L1, bcu),
                      SystemAssembler(a2, L2, bcp),
                      SystemAssembler(a3, L3, bcu)]

        # Apply bcs to matrices (this is done once)
        for a, A in zip(assemblers, As):
            a.assemble(A)

        # Chose between direct and iterative solvers
        solver_type = solver_params.get('la_solve', 'lu')
        assert solver_type in ('lu', 'la_solve')

        if solver_type == 'lu':
            solvers = map(lambda x: LUSolver("mumps"), range(3))
        else:
            solvers = [KrylovSolver('bicgstab', 'hypre_amg'),  # Very questionable preconditioner
                       KrylovSolver('cg', 'hypre_amg'),
                       KrylovSolver('cg', 'hypre_amg')]

        # Set matrices for once, likewise solver don't change in time
        for s, A in zip(solvers, As):
            s.set_operator(A)

            if solver_type == 'lu':
                s.parameters['reuse_factorization'] = True
            # Iterative tolerances
            else:
                s.parameters['relative_tolerance'] = 1E-8
                s.parameters['monitor_convergence'] = True

        gtime = 0.  # External clock

        # Things to remeber for evolution
        self.cylinder_bc_value = cylinder_bc_value
        # Keep track of time so that we can query it outside
        self.gtime, self.dt = gtime, dt
        # Remember inflow profile function in case it is time dependent
        self.inflow_profile = inflow_profile

        self.solvers = solvers
        self.assemblers = assemblers
        self.bs = bs
        self.u_, self.u_n = u_, u_n
        self.p_, self.p_n= p_, p_n

        # Rename u_, p_ for to standard names (simplifies processing)
        u_.rename('velocity', '0')
        p_.rename('pressure', '0')

        # Also expose measure for assembly of outputs outside
        self.ext_surface_measure = Measure('ds', domain=mesh, subdomain_data=surfaces)

        # Things to remember for easier probe configuration
        self.viscosity = mu
        self.density = rho
        self.normal = n
        self.cylinder_surface_tags = [cylinder_noslip_tag]

    def evolve(self, bc_value):
        '''Make one time step with the given values of jet boundary conditions'''
        assert len(bc_value) == 1
        self.cylinder_bc_value.A = bc_value[0]

        # Make a step
        self.gtime += self.dt(0)

        inflow = self.inflow_profile
        if hasattr(inflow, 'time'):
            inflow.time = self.gtime

        assemblers, solvers = self.assemblers, self.solvers
        bs = self.bs
        u_, p_ = self.u_, self.p_
        u_n, p_n = self.u_n, self.p_n

        solution_okay = True
        for (assembler, b, solver, uh) in zip(assemblers, bs, solvers, (u_, p_, u_)):
            assembler.assemble(b)
            try:
                solver.solve(uh.vector(), b)
            except:
                solution_okay = False

        solution_okay = solution_okay and not np.any(np.isnan(u_.vector().get_local()))
        solution_okay = solution_okay and not np.any(np.isnan(p_.vector().get_local()))

        if not solution_okay: warning('Simulation gone wrong')

        u_n.assign(u_)
        p_n.assign(p_)

        # Share with the world
        return u_, p_, solution_okay

# --------------------------------------------------------------------

if __name__ == '__main__':
    # This gives an evolved flow state from which control starts
    from generate_msh import generate_mesh
    from msh_convert import convert, cleanup
    import os
    
    # Start with mesh - goal is to crete ./mesh/mesh.h5
    geometry_params = {'output': 'mesh.geo',
                       'length': 2.2,
                       'front_distance': 0.05 + 0.15,
                       'bottom_distance': 0.05 + 0.15,
                       'jet_radius': 0.05,
                       'width': 0.41,
                       'jet_positions': [90, 270],
                       'jet_width': 10,
                       'clscale': 1.0}

    generate_mesh(geometry_params, template='geometry_2d.template_geo')
    # Convert to H5
    h5_path = './meshes/mesh.h5'
    (not os.path.exists(os.path.dirname(h5_path))) and os.mkdir(os.path.dirname(h5_path))
    
    convert('mesh.msh', h5_path)
    cleanup(exts=('.xml', '.msh'))
    # #    simulation_duration = 2.0 #duree en secondes de la simulation
    # #dt=0.0005

    # #root = 'mesh/turek_2d'
    # #if(not os.path.exists('mesh')):
    # #    os.mkdir('mesh')

    # #jet_angle = 0


    # def profile(mesh, degree):
    #     bot = mesh.coordinates().min(axis=0)[1]
    #     top = mesh.coordinates().max(axis=0)[1]
    #     print bot, top
    #     H = top - bot

    #     Um = 1.5

    #     return Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H',
    #                     '0'), bot=bot, top=top, H=H, Um=Um, degree=degree)

    # flow_params = {'mu': 1E-3,
    #               'rho': 1,
    #               'inflow_profile': profile}

    # solver_params = {'dt': dt}

    # list_position_probes = []

    # positions_probes_for_grid_x = [0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    # positions_probes_for_grid_y = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]

    # for crrt_x in positions_probes_for_grid_x:
    #     for crrt_y in positions_probes_for_grid_y:
    #         list_position_probes.append(np.array([crrt_x, crrt_y]))

    # positions_probes_for_grid_x = [-0.025, 0.0, 0.025, 0.05]
    # positions_probes_for_grid_y = [-0.15, -0.1, 0.1, 0.15]

    # for crrt_x in positions_probes_for_grid_x:
    #     for crrt_y in positions_probes_for_grid_y:
    #         list_position_probes.append(np.array([crrt_x, crrt_y]))

    # list_radius_around = [geometry_params['jet_radius'] + 0.02, geometry_params['jet_radius'] + 0.05]
    # list_angles_around = np.arange(0, 360, 10)

    # for crrt_radius in list_radius_around:
    #     for crrt_angle in list_angles_around:
    #         angle_rad = np.pi * crrt_angle / 180.0
    #         list_position_probes.append(np.array([crrt_radius * math.cos(angle_rad), crrt_radius * math.sin(angle_rad)]))

    # output_params = {'locations': list_position_probes,
    #                  'probe_type': 'pressure'
    #                  }

    # optimization_params = {"num_steps_in_pressure_history": 1,
    #                     "min_value_jet_MFR": -1e-2,
    #                     "max_value_jet_MFR": 1e-2,
    #                     "smooth_control": (nb_actuations/dt)*(0.1*0.0005/80),
    #                     "zero_net_Qs": True,
    #                     "random_start": random_start}

    # inspection_params = {"plot": plot,
    #                     "step": step,
    #                     "dump": dump,
    #                     "range_pressure_plot": [-2.0, 1],
    #                     "range_drag_plot": [-0.175, -0.13],
    #                     "range_lift_plot": [-0.2, +0.2],
    #                     "line_drag": -0.1595,
    #                     "line_lift": 0,
    #                     "show_all_at_reset": False,
    #                     "single_run":single_run
    #                     }

    # reward_function = 'drag_plain_lift'

    # verbose = 0

    # number_steps_execution = int((simulation_duration/dt)/nb_actuations)

    # # ---------------------------------------------------------------------------------
    # # do the initialization

    # #On fait varier la valeur de n-iter en fonction de si on remesh
    # if(remesh):
    #     n_iter = int(5.0 / dt)
    #     if(os.path.exists('mesh')):
    #         shutil.rmtree('mesh')
    #     os.mkdir('mesh')
    #     print("Make converge initial state for {} iterations".format(n_iter))
    # else:
    #     n_iter = None


    # #Processing the name of the simulation

    # simu_name = 'Simu'

    # if (geometry_params["jet_positions"][0] - 90) != 0:
    #     next_param = 'A' + str(geometry_params["jet_positions"][0] - 90)
    #     simu_name = '_'.join([simu_name, next_param])
    # if geometry_params["cylinder_size"] != 0.01:
    #     next_param = 'M' + str(geometry_params["cylinder_size"])[2:]
    #     simu_name = '_'.join([simu_name, next_param])
    # if optimization_params["max_value_jet_MFR"] != 0.01:
    #     next_param = 'maxF' + str(optimization_params["max_value_jet_MFR"])[2:]
    #     simu_name = '_'.join([simu_name, next_param])
    # if nb_actuations != 80:
    #     next_param = 'NbAct' + str(nb_actuations)
    #     simu_name = '_'.join([simu_name, next_param])
    # next_param = 'drag'
    # if reward_function == 'recirculation_area':
    #     next_param = 'area'
    # if reward_function == 'max_recirculation_area':
    #     next_param = 'max_area'
    # elif reward_function == 'drag':
    #     next_param = 'last_drag'
    # elif reward_function == 'max_plain_drag':
    #     next_param = 'max_plain_drag'
    # elif reward_function == 'drag_plain_lift':
    #     next_param = 'lift'
    # elif reward_function == 'drag_avg_abs_lift':
    #     next_param = 'avgAbsLift'
    # simu_name = '_'.join([simu_name, next_param])

    # env_2d_cylinder = Env2DCylinder(path_root=root,
    #                                 geometry_params=geometry_params,
    #                                 flow_params=flow_params,
    #                                 solver_params=solver_params,
    #                                 output_params=output_params,
    #                                 optimization_params=optimization_params,
    #                                 inspection_params=inspection_params,
    #                                 n_iter_make_ready=n_iter,  # On recalcule si besoin
    #                                 verbose=verbose,
    #                                 reward_function=reward_function,
    #                                 number_steps_execution=number_steps_execution,
    #                                 simu_name = simu_name)

    # return(env_2d_cylinder)

