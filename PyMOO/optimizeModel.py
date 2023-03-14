#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Wrapper to optimize a given ML model.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pymoo_helpers as helper
from optimizer import Optimizer
from problemWrapper import ProblemWrapper
from sklearn.metrics import r2_score
import myconfig as cfg

def main():

    ### ALGORITHM ###
    ### invoke a new instance of Optimizer ###
    myOptimizer = Optimizer()

    ### Define Algorithm for optimization ###
    myOptimizer.setAlgorithm('NSGA2')       # genetic algorithm
    print(myOptimizer.algorithm)


    ### PROBLEM ###
    """
    # name: Rosenbrock
    # n_var: 2
    # n_obj: 1
    # n_ieq_constr: 0
    # n_eq_constr: 0
    """
    n_var = 2
    n_obj = 1
    n_ieq_constr = 0
    n_eq_constr = 0
    xl = -2
    xu = 2

    myProblem = ProblemWrapper(
        n_var=n_var,  # hardcoded, but problem dependent
        n_obj=n_obj,  # hardcoded, but problem dependent
        n_ieq_constr=n_ieq_constr + 1,  # todo: was ist n_ieq_constr??
        xl=xl, xu=xu  # hardcoded, but problem dependent
    )

    # Provide ML model for problem wrapper
    myProblem.fetchPipelineFromFile()
    print("loaded model:", myProblem.pipeline)
    print("type of wrapped problem:", type(myProblem))

    # Provide problem for Optimizer
    myOptimizer.setProblem(myProblem)

    # Reduce algo iterations
    myOptimizer.setIterations( 500 )

    # Solve for optimal solution
    myOptimizer.solve(verbose=True)

    # Print solution
    res = myOptimizer.getSolution()
    result_stuff = [res.X, res.F, res.G, res.CV]
    print("Solution:")
    for r in result_stuff:
        print(r)


if __name__ == '__main__':
    main()



