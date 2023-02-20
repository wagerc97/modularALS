"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Example Usage from PyMOO documentation:
- We refer here to our documentation for all the details. However, for instance, executing NSGA2:
- The Zakharov function is a multi-objective optimization problem commonly used to test the performance of optimization 
  algorithms.

Note: 
- this project uses its own Python 3.10 interpreter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import utility as util


if __name__ == '__main__':

    # Define Problem
    problem = util.createProblem(problemName="rosenbrock")
    # SOO: Rosenbrock   (n_var: 2, n_obj: 1, n_constr: 0) -> solution is one single point
    # MOO: Zakharov     (n_var: 2, n_obj: 2, n_constr: 0) -> solution is pareto front in 2D

    # Define Algorithm
    algorithm = util.createAlgorithm(algo="nsga2")

    # Start the solver
    problem, res = util.mySolver(problem, algorithm, iterations=50)

    # Pretty print the solution
    util.summary(problem, res)

    # Plot the result
    util.plotResultWithPymoo(problem, res)
