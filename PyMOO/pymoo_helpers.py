#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Script with utility functions for mySolver.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os

import pandas as pd  # data handling
import numpy as np  # maths and stuff
import matplotlib.pyplot as plt     # plot stuff
from pymoo.optimize import minimize  # minimize solution
from pymoo.visualization.scatter import Scatter  # special pymoo plotter
from pymoo.visualization.fitness_landscape import FitnessLandscape  # allows illustrating problems as landscape

import myconfig



#### Common error with tutorial ####
# Error: Import error of get_problem
# Instead of from pymoo.problems import get_problem use from pymoo.problems.multi import * .

# And for get_problem use problem instead. As an example:
# > get_problem("zdt1").pareto_front()
# Should be converted to:
# > ZDT1().pareto_front()

######################################

def setup():
    print("\n")
    print("+" * 30)
    print(f"{__name__} is here")
    print("+" * 30, "\n")

def showFitnessLandscape(problem):
    """
    Show problem as landscape
    Illustrate problem (only applicable if one single objective function)
    """
    # 3D Surface model
    # Note: using n_samples to decrease the quality and allow turning 3D model on weak PC
    FitnessLandscape(problem, angle=(45, 45), _type="surface", n_samples=40, title=f"FitnessLandscape of {problem.name()}").show()
    # 2D Contour model
    #FitnessLandscape(problem, _type="contour", colorbar=True).show()


def createProblem(problem_name):
    """
    Create a problem
    :param problem_name: enter one { Zakharov/ZDT1, Rosenbrock }
    :return: the problem object
    """
    if problem_name.lower() in ("zakharov", "zdt1", "z"):
        from pymoo.problems.multi import ZDT1
        # https://pymoo.org/problems/single/zakharov.html
        problem = ZDT1(n_var=2)
    elif problem_name.lower() in ("rosenbrock", "r"):
        from pymoo.problems.single import Rosenbrock
        # https://pymoo.org/problems/single/rosenbrock.html
        problem = Rosenbrock(n_var=2)
        showFitnessLandscape(problem)

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


def solver(problem, algorithm, iterations, verbose=False):
    """
    Solve a problem with a chosen algorithm.
    :param problem: input problem
    :param algorithm: input algorithm
    :param iterations: number of steps towards optimal solution
    :param verbose: if True print solution in each iteration
    :return:
    """
    # Define Result
    # https://pymoo.org/interface/minimize.html
    res = minimize(problem,
                   algorithm,
                   ('n_gen', iterations),  # n_gen defines the number of iterations
                   verbose=verbose  # prints out solution in each iteration
                   )
    return problem, res


def summary(problem, res):
    """ Pretty print information about the solution of the result """
    NUM_SIGN = 78
    print("\n" + "-" * 30 + "[ RESULT SUMMARY ]" + "-" * 30)

    print(f"Elapsed time:\t{round(res.exec_time, 2)} seconds")
    print(f"Algorithm:\t\t{res.algorithm}")
    print(f"Problem:\t\t{problem.name()}")
    print(f"Result:\t\t\t{res}")
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


def writeResultValuesToFile(result):
    """
    Writes design space values (res.X aka X) and objective space values (res.F aka Y) to a csv file
    :param result:
    :return:
    """
    print("writing result values to file...")
    print("well actually not yet -> todo: implement this stuff ")


def plotResultWithPymoo(problem, result):
    """
    Plot the result.
    :param problem: the problem object
    :param result: the result object
    """
    plot = Scatter()
    plot.add(problem.pareto_front(),  # add line to indicate the pareto front
             plot_type="line",
             color="black",
             alpha=0.7)  # thickness
    plot.add(result.F,
             color="red")
    plot.show()


def plotOptimization(problem, res):
    """
    Plot the result.
    :param problem: the problem object
    :param res: the result object
    """
    val = res.algorithm
    plt.plot(np.arange(len(val)), val)
    plt.show()


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


def createRandomInputValue(problem, param_seed=42, N=10):
    """ Handles parameters for input data according to problem. """
    # Standard df size
    rows = N
    print("df rows:", rows)
    seed = param_seed
    print("random seed:", seed)

    # Problem dependent parameters
    if problem.name() in "Rosenbrock":
        lower = -2
        upper = 2
    elif problem.name() in "ZDT1":
        lower = -10
        upper = 10
    else:
        raise Exception("not implemented yet ")

    print("lower boundary:", lower)
    print("upper boundary:", upper)
    return generateDataframe(n_rows=rows, n_cols=problem.n_var,
                             x_lower=lower, x_upper=upper, seed=seed)


def computeOutputValues(train_x, problem):
    """
    Use pymoo problem to compute y for labels.
    :param train_x: input values in X
    :param problem: type of pymoo problem
    :return:
    """
    df = train_x.copy()     # always work on copy of df not to change original

    def evaluate_df_row(x):
        """
        Define a function to evaluate the problem for each row of the dataframe
        :param x: <class 'numpy.ndarray'>
        """
        return problem.evaluate(x)

    # Apply the evaluate_df_row function to each row of the dataframe
    results = np.apply_along_axis(evaluate_df_row, axis=1, arr=df.values)
    results_df = pd.DataFrame(results, columns=["y"])
    return results_df


def concatenateDataframes(x, y):
    """ Merge Dataframes """
    x_df = pd.DataFrame(x)
    frames = [x_df, y]
    combined_df = pd.concat(frames, axis=1)
    return combined_df


def storeDfInCsvFile(df, problem, deleteOldData):
    """
    Store df in csv file.
    :param df: combined dataframe
    :return:
    """
    #dirPath = os.path.join("..","data")
    #fileName = "data.csv"
    ##fullPath = dirPath+fileName+".csv"
    #fullPath = os.path.join(dirPath, fileName)
    #print("fullPath",fullPath)
    os.makedirs(myconfig.DATA_DIR, exist_ok=True)  # Make sure the directory exists
    # delete old train data file if True
    if deleteOldData:
        os.remove(myconfig.TRAIN_DATA_FILE)
    # write df to new csv file and delete old content
    df.to_csv(myconfig.TRAIN_DATA_FILE, encoding='utf-8',
              index=False,  # False: without index
              sep=";"  # custom seperator
              )
    # add problem name in first line
    with open(myconfig.TRAIN_DATA_FILE, "r+") as f:
        file_data = f.read()
        f.seek(0,0)     # get the first line
        f.write(str(problem.name()) + '\n' + file_data)

    print(f"Successfully stored {problem.name()}-data to location:", myconfig.TRAIN_DATA_FILE)
    return None


def plotData(df):
    # Extract the x and y data from the dataframe
    if 'x2' in df.columns:
        x1 = df['x1']
        x2 = df['x2']
        y = df['y']
    else:
        x1 = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    # Create the plot
    if 'x2' in df.columns:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, y)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
    else:
        fig, ax = plt.subplots()
        ax.plot(x1, y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    title = "New train data"+f" (n={df.shape[0]})"
    plt.title(title)
    plt.show()
    return None

