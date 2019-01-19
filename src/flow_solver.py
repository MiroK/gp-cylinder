from jet_bcs import JetBCValue
from utils import read_vtu_function
import numpy as np
from dolfin import *


class FlowSolver(object):
    '''IPCS scheme with explicit treatment of nonlinearity.'''
    def __init__(self, flow_params, geometry_params, solver_params):
        # Using very simple IPCS solver
        mu = Constant(flow_params['mu'])              # dynamic viscosity
        rho = Constant(flow_params['rho'])            # density

        mesh_file, surface_file = geometry_params['mesh']
        print 'One'
        # Load mesh with markers
        comm = mpi_comm_self()  # Don't share
        mesh = Mesh(comm, mesh_file)
        surfaces = MeshFunction('size_t', mesh, surface_file)
        print 'Two'
        # These tags should be hardcoded by gmsh during generation
        inlet_tag = 3
        outlet_tag = 2
        wall_tag = 1
        cylinder_noslip_tag = 4
        # Tags 5 and higher are jets

        # Define function spaces
        V = VectorFunctionSpace(mesh, 'CG', 2)
        Q = FunctionSpace(mesh, 'CG', 1)

        # Define trial and test functions
        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)

        # NOTE: VTK stores only P1 functions so
        u_p1, p_n = Function(VectorFunctionSpace(mesh, 'CG', 1)), Function(Q)

        # Starting from rest or are we given the initial state
        for path, func in zip(('u0_file', 'p0_file'), (u_p1, p_n)):
            if path in flow_params:
                foo = read_vtu_function(flow_params[path], func.function_space())[0]
                func.assign(foo)
        # Get P2 velocity
        u_n = project(u_p1, V)
        print 'Three'
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
        print 'Three_a'
        inflow_profile = flow_params.get('inflow_profile',
                                         FlowSolver.inflow_profile)(mesh, degree=2)
        print 'Three_aa'
        # Define boundary conditions, first those that are constant in time
        bcu_inlet = DirichletBC(V, inflow_profile, surfaces, inlet_tag)
        print 'Three_aaa'
        # No slip
        bcu_wall = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag)
        bcu_cyl_wall = DirichletBC(V, Constant((0, 0)), surfaces, cylinder_noslip_tag)
        # Fixing outflow pressure
        bcp_outflow = DirichletBC(Q, Constant(0), surfaces, outlet_tag)
        print 'Three_b'
        # Now the expression for the jets
        # NOTE: they start with Q=0
        radius = geometry_params['jet_radius']
        width = geometry_params['jet_width']
        positions = geometry_params['jet_positions']

        bcu_jet = []
        jet_tags = range(cylinder_noslip_tag+1, cylinder_noslip_tag+1+len(positions))

        print 'Three_c'
        jets = [JetBCValue(radius, width, theta0, Q=0, degree=1) for theta0 in positions]

        for tag, jet in zip(jet_tags, jets):
            bc = DirichletBC(V, jet, surfaces, tag)
            bcu_jet.append(bc)
        print 'Four'
        # All bcs objects togets
        bcu = [bcu_inlet, bcu_wall, bcu_cyl_wall] + bcu_jet
        bcp = [bcp_outflow]

        As = [Matrix(comm) for i in range(3)]
        bs = [Vector(comm) for i in range(3)]
        print 'Five'

        # Assemble matrices
        assemblers = [SystemAssembler(a1, L1, bcu),
                      SystemAssembler(a2, L2, bcp),
                      SystemAssembler(a3, L3, bcu)]
        print 'Six'
        # Apply bcs to matrices (this is done once)
        for a, A in zip(assemblers, As):
            a.assemble(A)
        print 'Seven'
        # Chose between direct and iterative solvers
        solver_type = solver_params.get('la_solve', 'lu')
        assert solver_type in ('lu', 'la_solve')

        if solver_type == 'lu':
            solvers = map(lambda x: LUSolver(comm, 'mumps'), range(3))
        else:
            assert False
        print 'Eight'
        # Set matrices for once, likewise solver don't change in time
        for s, A in zip(solvers, As):
            s.set_operator(A)
            s.parameters['reuse_factorization'] = True

        print 'Nine'
        gtime = 0.  # External clock

        # Things to remeber for evolution
        self.jets = jets
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
        self.cylinder_surface_tags = [cylinder_noslip_tag] + jet_tags
        print 'Done'
        
    def evolve(self, jet_bc_values):
        '''Make one time step with the given values of jet boundary conditions'''
        assert len(jet_bc_values) == len(self.jets)

        # Update bc expressions
        for Q, jet in zip(jet_bc_values, self.jets): jet.Q = Q

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

    @staticmethod
    def inflow_profile(mesh, degree):
        '''Parabolic with no slip on vertical'''
        bot = mesh.coordinates().min(axis=0)[1]
        top = mesh.coordinates().max(axis=0)[1]
        print 'XXXX'
        H = top - bot

        Um = 1.5

        #f = Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H',
        #                '0'), bot=bot, top=top, H=H, Um=Um, degree=degree)
        f = Constant((1, 0))

        print 'YYYYY'
        return f

# --------------------------------------------------------------------

if __name__ == '__main__':
    # This gives an evolved flow state from which control starts
    from generate_msh import generate_mesh
    from msh_convert import convert, cleanup
    import os, glob, shutil

    assert MPI.size(mpi_comm_world()) == 1
    
    # Start with mesh - goal is to crete ./mesh.xml
    mesh_path = './mesh/mesh.xml'
    surface_path = './mesh/mesh_facet_region.xml'
    
    geometry_params = {'output': 'mesh.geo',
                       'length': 2.2,
                       'front_distance': 0.05 + 0.15,
                       'bottom_distance': 0.05 + 0.15,
                       'jet_radius': 0.05,
                       'width': 0.41,
                       'jet_positions': [90, 270],
                       'jet_width': 10,
                       'clscale': 1.0}

    if not os.path.exists(mesh_path):
        generate_mesh(geometry_params, template='geometry_2d.template_geo')
        # Convert to H5
        not os.path.exists(os.path.dirname(mesh_path)) and os.mkdir(os.path.dirname(mesh_path))
    
        convert('mesh.msh', 'mesh.h5')
        cleanup(exts=('.msh', ))

        shutil.move('mesh.xml', mesh_path)
        shutil.move('mesh_facet_region.xml', surface_path)
        
    # With same dt and duration as with RL make the base flow
    dt = 0.0005
    solver_params = {'dt': dt}

    flow_params = {'mu': 1E-3,
                   'rho': 1}

    # Is there nonzero flow?
    u0_file, p0_file = 'mesh/u_init000000.vtu', 'mesh/p_init000000.vtu'
    if os.path.exists(u0_file) and os.path.exists(p0_file):
        flow_params['u0_file'] = u0_file
        flow_params['p0_file'] = p0_file
    
    geometry_params['mesh'] = (mesh_path, surface_path)

    solver = FlowSolver(flow_params, geometry_params, solver_params)
    print 'Initial state', solver.u_n.vector().norm('l2'), solver.p_n.vector().norm('l2')

    njets = len(geometry_params['jet_positions'])
    u_file = File('results/base_u.pvd')
    p_file = File('results/base_p.pvd')
    
    step = 0
    while solver.gtime < 0.2:
        step += 1
        uh, ph, status = solver.evolve(np.zeros(njets))
        
        # Plotting for debug
        if step % 40 == 0:
            u_file << uh, solver.gtime
            p_file << ph, solver.gtime
            print 'Solver time', solver.gtime, status
    # Final flow field which can be loaded
    File('./mesh/u_init.pvd') << uh
    File('./mesh/p_init.pvd') << ph
