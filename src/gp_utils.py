from deap import base, creator, tools, gp
import math, operator, random


def check_point_path(args):
    '''Recet population path based on cmd-line args'''
    args = args.__dict__
    
    path = ['%s:%s' % (k, args[k]) for k in ('cxpb', 'mutpb', 'ngen', 'popsize')]
    path = '_'.join(path)
    # Slot for ngen
    path = '_'.join([path, '%d.pkl'])

    return path


def history_path(args):
    '''Population fitness path based on cmd-line args'''
    args = args.__dict__
    
    path = ['%s:%s' % (k, args[k]) for k in ('cxpb', 'mutpb', 'ngen', 'popsize')]
    path = '_'.join(path)
    path = 'history_%s.txt' % path

    return path


def protectedDiv(arg0, arg1):
    '''Preclude division by zero'''
    if abs(arg1) > 1E-3: return arg0/arg1

    if 0 < abs(arg1) < 1E-3: return (arg1/abs(arg1))*arg0/1E-3

    return arg1/1E-3


def protectedLog(arg0):
    '''Preclude log negative'''
    if abs(arg0) > 1E-3: return (arg0/abs(arg0))*math.log(abs(arg0))

    if 0 < abs(arg0) < 1E-3: return (arg0/abs(arg0))*math.log(1E-3)

    return 0


def setup_GP_toolbox():
    '''Setup primitives, combinations, genetics for 4-variable expression'''
    # Register functions used to build the expression tree
    pset = gp.PrimitiveSet('MAIN', 4)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(abs, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.tanh, 1)
    pset.addPrimitive(protectedLog, 1)
    # Not really specified in the book, guess (-10, 10)
    pset.addEphemeralConstant("rand101", lambda: 10*(-1)**random.randrange(0, 2)*random.random())
    pset.renameArguments(ARG0='a1', ARG1='a2', ARG2='a3', ARG3='a4')

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Let the compule use the set of primitives
    toolbox.register("compile", gp.compile, pset=pset)
    
    # For these setting on actual mechanisom used to perform genetic operations
    # we'd need to look through the source code
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Limit complexity of the expression tree
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    toolbox.register("select", tools.selTournament, tournsize=7)  # See Ch 5.2.1 for tournament size

    return toolbox

# --------------------------------------------------------------------

if __name__ == '__main__':

    toolbox = setup_GP_toolbox()
    print toolbox.compile('sin(a1+a2)')(a1=1, a2=2, a3=3, a4=4)
