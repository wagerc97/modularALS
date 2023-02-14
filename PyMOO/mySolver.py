"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Example Usage from PyMOO documentation:
- We refer here to our documentation for all the details. However, for instance, executing NSGA2:
- The Zakharov function is a multi-objective optimization problem commonly used to test the performance of optimization 
  algorithms.

Note: 
- this project uses its own Python 3.10 interpreter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#### Common error with tutorial ####
# Error: Import error of get_problem
# Instead of from pymoo.problems import get_problem use from pymoo.problems.multi import * .

# And for get_problem use problem instead. As an example:
#> get_problem("zdt1").pareto_front()
# Should be converted to:
#> ZDT1().pareto_front()

######################################

import utility as util

if __name__ == '__main__':

    # Define Problem
    problem = util.createProblem(prob="r")
    # Rosenbrock(n_var: 2, n_obj: 2, n_constr: 0)
    # Zakharov(n_var: 2, n_obj: 2, n_constr: 0)

    # Define Algorithm
    algorithm = util.createAlgorithm(algo="nsga2")

    # Start the solver
    problem, res = util.mySolver(problem, algorithm, iterations=50)

    # Pretty print the solution
    util.summary(problem, res)

    # Plot the result
    util.plotResultWithPymoo(problem, res)
