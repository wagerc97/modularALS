"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Script with utility functions for mySolver.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd  # data handling
import numpy as np  # maths and stuff
from pymoo.optimize import minimize  # minimize solution
from pymoo.visualization.scatter import Scatter  # special pymoo plotter
from pymoo.visualization.fitness_landscape import FitnessLandscape  # allows illustrating problems as landscape
import MyCallback

#### Common error with tutorial ####
# Error: Import error of get_problem
# Instead of from pymoo.problems import get_problem use from pymoo.problems.multi import * .

# And for get_problem use problem instead. As an example:
# > get_problem("zdt1").pareto_front()
# Should be converted to:
# > ZDT1().pareto_front()

######################################


def showFitnessLandscape(problem):
    """
    Show problem as landscape
    Illustrate problem (only applicable if one single objective function)
    """
    FitnessLandscape(problem, angle=(45, 45), _type="surface").show()
    FitnessLandscape(problem, _type="contour", colorbar=True).show()


def createProblem(problemName):
    """
    Create a problem
    :param problemName: enter one { Zakharov/ZDT1, Rosenbrock }
    :return: the problem object
    """
    if problemName.lower() in ("zakharov", "zdt1", "z"):
        from pymoo.problems.multi import ZDT1
        # https://pymoo.org/problems/single/zakharov.html
        problem = ZDT1(n_var=2)
    elif problemName.lower() in ("rosenbrock", "r"):
        from pymoo.problems.single import Rosenbrock
        # https://pymoo.org/problems/single/rosenbrock.html
        problem = Rosenbrock(n_var=2)
        # showFitnessLandscape(problem)

    else:
        raise Exception("Enter parameter { Zakharov, Rosenbrock }")
    print(problem)
    return problem


def createAlgorithm(algo):
    """
    Create an algorithm
    :param algo: enter one { NSGA2 }
    :return: the algorithm object
    """
    if algo.lower() == "nsga2" or "n":
        from pymoo.algorithms.moo.nsga2 import NSGA2
        algorithm = NSGA2(pop_size=100)  # pop_size defines the number of dots

    else:
        raise Exception("Enter parameter { NSGA2 }")

    return algorithm


def mySolver(problem, algorithm, iterations):
    """
    Create and solve a problem with a chosen algorithm
    :param problem: input problem
    :param algorithm: input algorithm
    :param iterations: number of steps towards optimal solution
    :return:
    """
    newMyCallback = MyCallback()
    # Define Result
    # https://pymoo.org/interface/minimize.html
    res = minimize(problem,
                   algorithm,
                   ('n_gen', iterations),  # n_gen defines the number of iterations
                   verbose=True,  # prints out solution in each iteration
                    callback=newMyCallback
                   )
    return problem, res


def summary(problem, res):
    """ Pretty print information about the solution of the result """
    NUM_SIGN = 80
    print("\n" + "-" * 31 + "[ RESULT SUMMARY ]" + "-" * 31)

    print(f"Elapsed time:\t{round(res.exec_time, 2)} seconds")
    print(f"Algorithm:\t\t{res.algorithm}")
    print(f"Problem:\n{problem.name}")
    print(f"Result:\n{res}")
    print("")

    printResult = False
    if printResult:
        # Create DataFrames for decision variables and objectives and print them
        X_df = pd.DataFrame(res.X, columns=[f"x{i + 1}" for i in range(problem.n_var)])
        F_df = pd.DataFrame(res.F, columns=[f"f{i + 1}" for i in range(problem.n_obj)])
        print("Decision variables:")
        print(X_df)
        print("\nObjectives:")
        print(F_df)

    if True:
        writeResultValuesToFile(res)

    print("-" * NUM_SIGN)


def writeResultValuesToFile(res):
    """
    Writes design space values (res.X aka X) and objective space values (res.F aka Y) to a csv file
    :param res:
    :return:
    """
    print("writing result values to file...")


def plotResultWithPymoo(problem, res):
    """
    Plot the result.
    :param problem: the problem object
    :param res: the result object
    """
    plot = Scatter()
    plot.add(problem.pareto_front(),  # add line to indicate the pareto front
             plot_type="line",
             color="black",
             alpha=0.7)  # thickness
    plot.add(res.F,
             color="red")
    plot.show()


def generateDataframe(n_rows, n_cols, x_lower, x_upper, seed=42):
    """
    Generate a dataframe with random numbers. Parameters define dimensions and boundaries.
    :param n_rows: ROWS - df length
    :param n_cols: COLS - number of columns
    :param x_lower: lower bound
    :param x_upper: upper bound
    :param seed: random seed
    :return: resulting dataframe
    """
    np.random.seed(seed)  # Set the random seed to 42
    df_dict = {}  # dataframe dictionary will store the columns temporarily
    for i in range(1, n_cols + 1):
        print(f"randomly draw column: x{i}")
        # Generate n random numbers between the specified lower and upper bounds using the uniform() function
        x = np.random.uniform(low=x_lower, high=x_upper, size=n_rows)
        # add new column to dataframe dictionary
        df_dict[f'x{i}'] = x

    df = pd.DataFrame(df_dict)  # convert dict to df
    return df


def createRandomInputValue(problem):
    """ Handels parameters for input data according to problem. """
    # Standard df size
    rows = 10
    print("df rows:", rows)
    seed = 42
    print("random seed:", seed)

    # Problem dependent parameters
    if problem.name().lower() in "rosenbrock":
        lower = -2
        upper = 2
    elif problem.name().lower() in "zakharov":
        lower = -2
        upper = 2
    else:
        raise Exception("not implemented yet ")

    print("lower boundary:", lower)
    print("upper boundary:", upper)
    return generateDataframe(n_rows=rows,
                             n_cols=problem.n_var,
                             x_lower=lower,
                             x_upper=upper,
                             seed=seed
                             )


def computeOutputValues(train_x, problem):
    labels = []  # list for labels
    for x in train_x:
        y = problem(x)
        print(y)
        labels.append(y)
    return labels
