# Optimal control of flow past a cylinder with genetic programming

The problem is identical to [Rabault et al](https://arxiv.org/abs/1808.07664) but the optimization step uses GP.
Solver instructure is taken from paper's code [base](https://github.com/jerabaul29/Cylinder2DFlowControlDRL).

## Dependencies
- FEniCS stack
- deap
- gmsh

## TODO
- threaded and parallel search; parallel has MPI_COMM_SELF