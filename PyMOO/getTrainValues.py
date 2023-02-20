"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Compute and store X-Y value pairs from different problems. 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import utility as util


if __name__ == '__main__':

    # Define Problem
    problem = util.createProblem(prob="rosenbrock")
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
