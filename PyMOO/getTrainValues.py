"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Compute and store X-Y value pairs from different problems. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import utility as util

if __name__ == '__main__':

    # Chose a problem
    # - SOO: Rosenbrock (x1, x2, y)      (n_var: 2, n_obj: 1, n_constr: 0) -> solution is one single point
    # - MOO: Zakharov   (x1, x2, y1, y2) (n_var: 2, n_obj: 2, n_constr: 0) -> solution is pareto front in 2D
    PROBLEM_NAME = "rosenbrock"

    # Define Problem
    problem = util.createProblem(problemName=PROBLEM_NAME)

    # Define random input values
    train_x = util.createRandomInputValue(problem)
    print("\nNew Dataframe for X:\n", train_x)

    # Compute according output values
    train_y = util.computeOutputValues(train_x, problem)
    print("\nNew Dataframe for Y:\n", train_y)

    # Plot random values
    # ... function()


