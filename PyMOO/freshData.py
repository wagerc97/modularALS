#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Compute and store X-Y value pairs from different problems. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# import pymoo_helpers as helper
import os
import sys

import pandas as pd  # data handling
import numpy as np  # maths and stuff
import matplotlib.pyplot as plt  # plot stuff
from pymoo.optimize import minimize  # minimize solution
from pymoo.visualization.scatter import Scatter  # special pymoo plotter
from pymoo.visualization.fitness_landscape import FitnessLandscape  # allows illustrating problems as landscape

import myconfig


class FreshData:
    def __init__(self, n, problem_name, seed=42, deleteOldData=True, algorithm_name='nsga2', **kwargs):
        """
        A class for generating, expanding and storing new data for training and prediction.

        # Chose a problem
        # - SOO: Rosenbrock (x1, x2, y)      (n_var: 2, n_obj: 1, n_constr: 0) -> solution is one single point
        # - MOO: Zakharov   (x1, x2, y1, y2) (n_var: 2, n_obj: 2, n_constr: 0) -> solution is pareto front in 2D
        """
        self.n = n
        self.seed = seed
        self.deleteOldData = deleteOldData
        self.problem_name = problem_name
        self.problem = None
        self.algorithm_name = algorithm_name  # nsga2 is default
        self.algorithm = None
        self.iterations = 100
        self.result = None  # computed datapoints for result
        self.upper = None
        self.lower = None
        self.X = None  # input matrix
        self.y = None  # labels
        self.df = None  # combined dataframe [ X | y ]


    def createProblem(self):
        """
        Create a problem
        :param problem_name: enter one { Zakharov/ZDT1, Rosenbrock }
        :return: the problem object
        """
        if self.problem_name.lower() in ("zakharov", "zdt1", "z"):
            from pymoo.problems.multi import ZDT1
            # https://pymoo.org/problems/single/zakharov.html
            self.problem = ZDT1(n_var=2)
        elif self.problem_name.lower() in ("rosenbrock", "r"):
            from pymoo.problems.single import Rosenbrock
            # https://pymoo.org/problems/single/rosenbrock.html
            self.problem = Rosenbrock(n_var=2)
            self.showFitnessLandscape()

        else:
            raise Exception("Enter parameter { Zakharov, Rosenbrock }")
        print(self.problem)


    def showFitnessLandscape(self):
        """
        Show problem as landscape
        Illustrate problem (only applicable if one single objective function)
        """
        # 3D Surface model
        # Note: using n_samples to decrease the quality and allow turning 3D model on weak PC
        FitnessLandscape(self.problem, angle=(45, 45), _type="surface", n_samples=40,
                         title=f"FitnessLandscape of {self.problem.name()}").show()
        # 2D Contour model
        # FitnessLandscape(problem, _type="contour", colorbar=True).show()


    def createAlgorithm(self, algo=None):
        """
        Create an algorithm
        :param algo: enter one { NSGA2 }
        :return: the algorithm object
        """
        if algo is None:
            algo = self.algorithm_name

        if algo.lower() == "nsga2" or "n":
            from pymoo.algorithms.moo.nsga2 import NSGA2
            self.algorithm = NSGA2(pop_size=100)  # pop_size defines the number of dots
        else:
            raise Exception("Enter parameter { NSGA2 }")


    def solver(self, iterations=None):
        """
        Create and solve a problem with a chosen algorithm
        :param problem: input problem
        :param algorithm: input algorithm
        :param iterations: number of steps towards optimal solution
        :return:
        """
        if iterations is None:
            iterations = self.iterations

        # Define Result
        # https://pymoo.org/interface/minimize.html
        self.result = minimize(self.problem,
                               self.algorithm,
                               ('n_gen', iterations),  # n_gen defines the number of iterations
                               verbose=True  # prints out solution in each iteration
                               )


    def summary(self):
        """ Pretty print information about the solution of the result """
        NUM_SIGN = 80
        print("\n" + "-" * 31 + "[ RESULT SUMMARY ]" + "-" * 31)

        print(f"Elapsed time:\t{round(self.result.exec_time, 2)} seconds")
        print(f"Algorithm:\t\t{self.result.algorithm}")
        print(f"Problem:\t\t{self.problem.name()}")
        print(f"Result:\t\t{self.result}")
        print("")

        printResult = False
        if printResult:
            # Create DataFrames for decision variables and objectives and print them
            X_df = pd.DataFrame(self.result.X, columns=[f"x{i + 1}" for i in range(self.problem.n_var)])
            F_df = pd.DataFrame(self.result.F, columns=[f"f{i + 1}" for i in range(self.problem.n_obj)])
            print("Decision variables:")
            print(X_df)
            print("\nObjectives:")
            print(F_df)
        print("-" * NUM_SIGN)


    def setBoundaries(self, verbose=False):
        """ Handles parameters for input data according to problem. """
        # Standard df size
        if verbose:
            print("df rows:", self.n)
            print("random seed:", self.seed)
        # Problem dependent parameters
        if self.problem.name() in "Rosenbrock":
            self.lower = -2
            self.upper = 2
        elif self.problem.name() in "ZDT1":
            self.lower = -10
            self.upper = 10
        else:
            raise Exception("not implemented yet ")
        if verbose:
            print("lower boundary:", self.lower)
            print("upper boundary:", self.upper)


    def generateRandomNumberDataframe(self):
        """ Generate a dataframe with random numbers. Parameters define dimensions and boundaries. """
        self.setBoundaries()

        if self.problem is None:
            raise Exception("Error: Define problem first, then generate random data")
        else:
            n_cols = self.problem.n_var

        np.random.seed(self.seed)  # Set the random seed to 42
        df_dict = {}  # dataframe dictionary will store the columns temporarily
        for i in range(1, n_cols + 1):
            print(f"randomly draw column: x{i}")
            # Generate n random numbers between the specified lower and upper bounds using the uniform() function
            x = np.random.uniform(low=self.lower, high=self.upper, size=self.n)
            # add new column to dataframe dictionary
            df_dict[f'x{i}'] = x

        self.X = pd.DataFrame(df_dict)  # convert dict to df


    def computeOutputValues(self):
        """ Use pymoo problem to compute y for labels. """


        def evaluate_df_row(x):
            """
            Define a function to evaluate the problem for each row of the dataframe
            :param x: <class 'numpy.ndarray'>
            """
            return self.problem.evaluate(x)

        # Apply the evaluate_df_row function to each row of the dataframe
        results = np.apply_along_axis(evaluate_df_row, axis=1, arr=self.X.values)
        self.y = pd.DataFrame(results, columns=["y"])


    def concatenateDataframes(self, x, y):
        """ Merge Dataframes """
        x_df = pd.DataFrame(x)
        frames = [x_df, y]
        combined_df = pd.concat(frames, axis=1)
        self.df = combined_df


    def storeDfInCsvFile(self):
        """ Store df in csv file """
        # dirPath = os.path.join("..","data")
        # fileName = "data.csv"
        ##fullPath = dirPath+fileName+".csv"
        # fullPath = os.path.join(dirPath, fileName)
        # print("fullPath",fullPath)
        os.makedirs(myconfig.DATA_DIR, exist_ok=True)  # Make sure the directory exists
        # delete old train data file if True
        if self.deleteOldData:
            os.remove(myconfig.DATA_FILE)
        # write df to new csv file and delete old content
        self.df.to_csv(myconfig.DATA_FILE, encoding='utf-8',
                       index=False,  # False: without index
                       sep=";"  # custom seperator
                       )
        # add problem name in first line
        with open(myconfig.DATA_FILE, "r+") as f:
            file_data = f.read()
            f.seek(0, 0)  # get the first line
            f.write(str(self.problem.name()) + '\n' + file_data)

        print(f"Successfully stored {self.problem.name()}-data to location:", myconfig.DATA_FILE)


if __name__ == '__main__':
    main()
