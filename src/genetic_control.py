# Employ genetic programming to control the mean field problem
from eval_control import ControlEvaluator, VTKIO, NumpyIO
import operator, math, random, sys
from mpi4py import MPI
import numpy as np
# The idea here to most config of flow solver except for IO which will
# depend on the expression that is the control

# Flow solver params
geometry_params = {'jet_radius': 0.05,
                   'jet_positions': [90, 270],
                   'jet_width': 10,
                   'mesh': ('./mesh/mesh.xml', './mesh/mesh_facet_region.xml')}

solver_params = {'dt': 5E-4}

flow_params = {'mu': 1E-3,
               'rho': 1,
               'u0_file': './mesh/u_init000000.vtu',
               'p0_file': './mesh/p_init000000.vtu'}

# Optimization
angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
# Kick the front and back
angles = angles[[2, 3, 7, 8]]
radius = geometry_params['jet_radius']
probe_positions = np.c_[radius*np.cos(angles), radius*np.sin(angles)]

optimization_params = {'T_term': 0.1/4,
                       'dt_control': 20*solver_params['dt'],
                       'smooth_control': 0.1,
                       'probe_positions': probe_positions}

params = {'geometry': geometry_params,
          'flow': flow_params,
          'solver': solver_params,
          'optim': optimization_params}

controller = ControlEvaluator(params)


def eval_control_config(control, toolbox, params=params, controller=controller):
    '''Eval using toolbox compiled control -> 1 tuple'''
    # Each cpu runs different
    hash_control = hash(control)
    # -121243 is not a good name for file
    if hash_control < 0:
        hash_control = -hash_control*10

    vtk_io = None #VTKIO('./results/test/%s' % hash_control, ['uh', 'ph'], 1)
    np_io = NumpyIO('./results/test/%s.txt' % hash_control, 20)

    ctrl = toolbox.compile(expr=control)
    # Make it expression of time as well and compile eventually
    def f(t, a1, a2, a3, a4, ctrl=ctrl):
        v = ctrl(a1, a2, a3, a4)
        return min(0.01, v) if v > 0 else max(-0.01, v)

    # Decisions about cost should be made here
    score, evolve_okay, (t, T) = controller.eval_control(f, 
                                                         history=None, vtk_io=vtk_io, np_io=np_io)
    # Solver okay and we reached terminal
    if evolve_okay and t > T: return (score, )

    # Others are weighted by relative distance to terminal
    return (score*math.exp(abs((t - T)*T)), )
    
# TODO
# - reward
# - solver and eval params as in RL
# - capping
# - make sure that init okay
# - primitives


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse, os, pickle, sys, multiprocessing, itertools, time
    from gp_utils import check_point_path, history_path, setup_GP_toolbox
    from mpi_thread import distribute, collect
    from deap import algorithms, tools

    comm = MPI.COMM_WORLD
    
    # An argument can be passed that points to the file with stored best law
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Which module to test?
    parser.add_argument('-check_point', type=str, default='', help='Path to file with dumped state')
    parser.add_argument('-cp_freq', type=int, default=2, help='Check point frequency')
    parser.add_argument('-dir_name', type=str, default='results', help='How to name checkpoint folder')

    # Start or continue search
    parser.add_argument('-cxpb', type=float, default=0.6, help='Crossover probability')
    parser.add_argument('-mutpb', type=float, default=0.3, help='Mutation probability')
    parser.add_argument('-ngen', type=int, default=4, help='Number of generations')
    parser.add_argument('-popsize', type=int, default=8, help='Number of individuals in generation')

    # Eval the loaded state
    plot_parser = parser.add_mutually_exclusive_group(required=False)
    plot_parser.add_argument('--plot', dest='do_plot', action='store_true')
    plot_parser.add_argument('--no_plot', dest='do_plot', action='store_false')
    parser.set_defaults(do_plot=False)

    args = parser.parse_args()

    # -----------------
    
    state_file_template = check_point_path(args)
    state_file_template = os.path.join(args.dir_name, state_file_template)

    history_path = history_path(args)

    if comm.rank == 0:
        not os.path.exists(args.dir_name) and os.mkdir(args.dir_name)

    # Toolbox for compilation is needed everywhere
    toolbox = setup_GP_toolbox()

    # FIXME
    def fitness(indiv, toolbox=toolbox):
        '''Fitness of the individual'''
        return eval_control_config(indiv, toolbox)
    
    # # Look at results
    # # FIXME
    # # if args.do_plot and comm.rank == 0:
    # #     assert args.check_point
    # #     # FIXME
    # #     data, cost = eval_best_control(args.check_point, toolbox, args.dt)
    # #     plot_results(data)

    # #     with open(args.check_point, "r") as cp_file:
    # #         cp = pickle.load(cp_file)
    # #     logbook = cp['logbook']

    # #     gen = int((args.check_point.split('_')[-1]).split('.')[0])

    # #     nevals = [r['nevals'] for r in logbook]
    # #     np.savetxt(('nevals@_%d' % gen )+ history_path, nevals)

    # #     sys.exit(0)

    # Learn
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    seed = 123
    random.seed(123)

    # Root loads
    if comm.rank == 0:
        if not args.check_point:
            population = toolbox.population(n=args.popsize)  
            halloffame = tools.HallOfFame(1)

            start_gen = 0
            logbook = tools.Logbook()
        else:
            with open(args.check_point, "r") as cp_file: cp = pickle.load(cp_file)

            population = cp["population"]
            halloffame = cp["halloffame"]

            start_gen = cp["generation"]
            logbook = cp["logbook"]
            random.setstate(cp["rndstate"])

        pop_history, best_history = [], []
    else:
        start_gen = None

    start_gen = comm.bcast(start_gen, 0)

    # Local workers
    current_min = sys.float_info.max
    msg = 'Rank %d processed %d individuals in %g s'
    # Actual search
    for gen in range(start_gen, args.ngen):
        # Root breads
        if comm.rank == 0:
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, args.cxpb, args.mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # Turn to string programs and distribute
        my_individuals, _ = distribute(map(str, invalid_ind) if comm.rank == 0 else None, comm)

        # Locally eval their fitness
        t0 = time.time()

        fitnesses = map(fitness, my_individuals)
        dt = time.time()-t0
        nindivs = len(fitnesses)
        
        print msg % (comm.rank, nindivs, dt)
        
        # Collect on root
        fitnesses = collect(np.array(fitnesses, dtype=float), comm)
        # Root breeds:
        if comm.rank == 0:
            
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit, )

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)
                current_min, = halloffame.keys[0].getValues()
                best_history.append(current_min)

                np.savetxt('best_' + history_path, best_history)
            # Replace the current population by the offspring
            population[:] = offspring

            # Save all fitness for histogram etc
            pop_history.append(
                [p.fitness.getValues()[0] for p in population]
            )
            np.savetxt('pop_' + history_path, pop_history)

            # Append the current generation statistics to the logbook
            record = mstats.compile(population) if mstats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # Keep track of number of evaluations
            nevals = [r['nevals'] for r in logbook]
            np.savetxt('nevals_' + history_path, nevals)

            print logbook.stream

            if gen % args.cp_freq == 0 or gen == (args.ngen-1):
                cp = dict(population=population, 
                          generation=gen,
                          halloffame=halloffame,
                          logbook=logbook,
                          rndstate=random.getstate())
            
                state_file = state_file_template % gen
                with open(state_file, "wb") as cp_file:
                    pickle.dump(cp, cp_file, -1)
                    
        # Let root tell everybody about current_min
        current_min = comm.bcast(current_min, 0)
