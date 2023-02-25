#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Compute and store X-Y value pairs from different problems. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import utility as util
import numpy as np

if __name__ == '__main__':

    # Chose a problem
    # - SOO: Rosenbrock (x1, x2, y)      (n_var: 2, n_obj: 1, n_constr: 0) -> solution is one single point
    # - MOO: Zakharov   (x1, x2, y1, y2) (n_var: 2, n_obj: 2, n_constr: 0) -> solution is pareto front in 2D
    PROBLEM_NAME = "Rosenbrock"
    SEED = 42
    N = 300

    # Define Problem
    problem = util.createProblem(problemName=PROBLEM_NAME)

    # Define random input values
    train_x = util.createRandomInputValue(problem, SEED, N)
    print("\nX:\n", train_x)

    # Compute according output values
    train_y = util.computeOutputValues(train_x, problem)
    print("\ny:\n", train_y)

    # Merge Dataframes
    df = util.concatenateDataframes(train_x, train_y)
    print("\nCombined df:\n", df)

    # Store df in csv file
    util.storeDfInCsvFile(df, problem)

    # Plot random values
    util.plotData(df)


