#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Example Usage from PyMOO documentation:
- We refer here to our documentation for all the details. However, for instance, executing NSGA2:
- The Zakharov function is a multi-objective optimization problem commonly used to test the performance of optimization 
  algorithms.

Note: 
- this project uses its own Python 3.10 interpreter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import sys, os
sys.path.append(os.path.abspath(os.path.join('.', 'PYMOO', 'test')))  # '.' for main in .../Optimization/main.py

import PyMOO.pymoo_helpers as helper


if __name__ == '__main__':

    # Define Problem
    problem = helper.createProblem(problem_name="rosenbrock")
    # SOO: Rosenbrock   (n_var: 2, n_obj: 1, n_constr: 0) -> solution is one single point
    # MOO: Zakharov     (n_var: 2, n_obj: 2, n_constr: 0) -> solution is pareto front in 2D

    # Define Algorithm
    algorithm = helper.createAlgorithm(algo="nsga2")

    # Start the solver
    problem, res = helper.solver(problem, algorithm, iterations=50)

    # print result
    result_stuff = [res.X, res.F, res.G, res.CV]
    print("Solution:")
    for r in result_stuff:
        print(r)


    # Pretty print the solution
    helper.summary(problem, res)

    # Plot the result
    helper.plotResultWithPymoo(problem, res)
