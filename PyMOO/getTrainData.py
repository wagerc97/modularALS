#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Compute and store X-Y value pairs from different problems. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pymoo_helpers as helper

# Chose a problem
# - SOO: Rosenbrock (x1, x2, y)      (n_var: 2, n_obj: 1, n_constr: 0) -> solution is one single point
# - MOO: Zakharov   (x1, x2, y1, y2) (n_var: 2, n_obj: 2, n_constr: 0) -> solution is pareto front in 2D
PROBLEM_NAME = "Rosenbrock"
SEED = 42
N = 300

def main(n=N, seed=SEED, problem_name=PROBLEM_NAME, deleteOldData=False):

    helper.setup()

    # Define Problem
    problem = helper.createProblem(problem_name=problem_name)

    # Define random input values
    train_x = helper.createRandomInputValue(problem, seed, n)
    print("\nX:\n", train_x)

    # Compute according output values
    train_y = helper.computeOutputValues(train_x, problem)
    print("\ny:\n", train_y)

    # Merge Dataframes
    df = helper.concatenateDataframes(train_x, train_y)
    print("\nCombined df:\n", df)

    # Store df in csv file
    helper.storeDfInCsvFile(df, problem, deleteOldData)

    # Plot random values
    #helper.plotData(df)


if __name__ == '__main__':
    main()
