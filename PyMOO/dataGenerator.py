#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Compute and save new data matrices from different problems of the form 
[ x1 | x2 |...| y ] 
Stored as CSV files. See myconfig for specific file location.
File location is handled via default filepaths stored in myconfig.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# import pymoo_helpers as helper
import os
import pandas as pd  # data handling
import numpy as np  # maths and stuff
import matplotlib.pyplot as plt  # plot stuff
from pymoo.visualization.scatter import Scatter  # special pymoo plotter
from pymoo.optimize import minimize  # minimize solution
from pymoo.visualization.fitness_landscape import FitnessLandscape  # allows illustrating problems as landscape
import myconfig


class DataGenerator:
    def __init__(self, mode=None, n=100, problem_name="Rosenbrock", seed=42, overwrite=True, **kwargs):
        """
        A class for generating, expanding and storing new data for training and prediction.
        # Chose a problem
        # - SOO: Rosenbrock (x1, x2, y)      (n_var: 2, n_obj: 1, n_constr: 0) -> solution is one single point
        # - MOO: Zakharov   (x1, x2, y1, y2) (n_var: 2, n_obj: 2, n_constr: 0) -> solution is pareto front in 2D
        """
        self.mode = None                        # defines what kind of data will be generated
        self.csvFileName = None                 # file name of resulting csv file
        self.csvFilePath = None                 # file path of resulting csv file
        self.n = n                              # number of data points
        self.seed = seed                        # random seed
        self.overwrite = overwrite              # Boolean: if True overwrite old data csv file
        self.problem_name = problem_name        # name of problem given as parameter
        self.problem = None                     # Pymoo problem object
        self.result = None                      # computed datapoints for result
        self.upper = None                       # upper boundary
        self.lower = None                       # lower boundary
        self.X = None                           # input matrix
        self.y = None                           # labels
        self.df = None                          # combined dataframe [ X | y ]

        # Validation
        self._validateMode(mode)                # set self.mode according to given mode parameter
        self.setProblem()                       # initially sets a problem

        print(f"New {self.__class__.__name__} object with mode='{self.mode}' created ")


    def __del__(self):
        """ destructor frees up memory """
        print(f"\nObject {self.__class__.__name__} destroyed")


    def _validateMode(self, param_mode, verbose=True):
        """
        Set self.mode according to given mode parameter.
        Can be changed later on
        """
        if param_mode in ("t", "train", "training"):
            self.mode = myconfig._modeTRAIN
            self.csvFileName = myconfig.TRAIN_DATA_NAME
            self.csvFilePath = myconfig.TRAIN_DATA_FILE
        elif param_mode in ("p", "pred", "predict", "prediction", "predicting"):
            self.mode = myconfig._modePREDICT
            self.csvFileName = myconfig.PRED_DATA_NAME
            self.csvFilePath = myconfig.PRED_DATA_FILE
            self.seed += 1  # alter the random seed to generate new unseen data
        elif self.mode is None or self.mode == "":
            raise TypeError("TypeError: DataGenerator.__init__() missing 1 required positional argument: 'mode'!\nValid modes are {'train' or 'predict'}")
        else:
            raise ValueError(f"DataGenerator received invalid mode: '{param_mode}'. Valid modes are ['train' or 'predict']")
        if verbose:
            print("csvFileName:", self.csvFileName)
            print("csvFilePath:", self.csvFilePath)
            print("seed:", self.seed)


    def setProblem(self, plotProblem=False):
        if self.problem_name.lower() in ("zakharov", "zdt1", "z"):
            from pymoo.problems.multi import ZDT1
            # https://pymoo.org/problems/single/zakharov.html
            self.problem = ZDT1(n_var=2)
        elif self.problem_name.lower() in ("rosenbrock", "r"):
            from pymoo.problems.single import Rosenbrock
            # https://pymoo.org/problems/single/rosenbrock.html
            self.problem = Rosenbrock(n_var=2)
            if plotProblem:
                self.showFitnessLandscape()

        else:
            raise ValueError("Enter one problem_name of { Zakharov, Rosenbrock }")
        print(self.problem)
        # Set boundary values according to problem
        self.setBoundaries()


    def getProblem(self):
        return self.problem


    def getProblemParameters(self):
        return self.problem.n_var, self.problem.n_obj, self.problem.n_ieq_constr, self.lower, self.upper


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


    def summary(self, printResult=False):
        """ Pretty print information about the solution of the result """
        if self.result is None:
            raise ValueError("Error: You need to compute the problem result first")

        else:
            NUM_SIGN = 80
            print("\n" + "-" * 31 + "[ RESULT SUMMARY ]" + "-" * 31)

            print(f"Elapsed time:\t{round(self.result.exec_time, 2)} seconds")
            print(f"Algorithm:\t\t{self.result.algorithm}")
            print(f"Problem:\t\t{self.problem.name()}")
            print(f"Result:\t\t{self.result}")
            print("")
        try:
            if printResult:
                # Create DataFrames for decision variables and objectives and print them
                X_df = pd.DataFrame(self.result.X, columns=[f"x{i + 1}" for i in range(self.problem.n_var)])
                F_df = pd.DataFrame(self.result.F, columns=[f"f{i + 1}" for i in range(self.problem.n_obj)])
                print("Decision variables:")
                print(X_df)
                print("\nObjectives:")
                print(F_df)
        except Exception as e:
            print("Could not print results in dataGenerator.summary(). Here is why:\n", str(e))
        print("-" * NUM_SIGN)


    def generateRandomX(self):
        """ Generate a dataframe with random numbers. """
        if self.problem is None:
            raise Exception("Error: Define problem first, then generate random data")
        else:
            n_cols = self.problem.n_var
        np.random.seed(self.seed)  # Set the random seed to 42
        df_dict = {}  # dataframe dictionary will store the columns temporarily
        for i in range(1, n_cols + 1):
            print(f"randomly draw numbers for problem input (column x{i})")
            # Generate n random numbers between the specified lower and upper bounds using the uniform() function
            x = np.random.uniform(low=self.lower, high=self.upper, size=self.n)
            # add new column to dataframe dictionary
            df_dict[f'x{i}'] = x
        self.X = pd.DataFrame(df_dict)  # convert dict to df


    def get_X(self):
        if self.X is None:
            raise ValueError("Error: First you have to define X data using f.e. generateRandomX()")
        return self.X


    def concatenateDataframes(self, x, y):
        """ Merge Dataframes and store in new dataframe to use with ml model """
        x_df = pd.DataFrame(x)
        frames = [x_df, y]
        combined_df = pd.concat(frames, axis=1)
        self.df = combined_df


    def evaluate_df_row(self, x):
        """
        Define a function to evaluate the problem for each row of the dataframe
        :param x: <class 'numpy.ndarray'>
        """
        return self.problem.evaluate(x)

    def computeLabels(self):
        """ Use pymoo problem to compute y for labels. """
        # Apply the evaluate_df_row function to each row of the dataframe
        results = np.apply_along_axis(self.evaluate_df_row, axis=1, arr=self.X.values)
        self.y = pd.DataFrame(results, columns=["y"])
        # Merge and store the two dataframes as one to use with ml model
        self.concatenateDataframes(self.X, self.y)


    def get_y(self):
        if self.y is None:
            raise ValueError("Error: First you have to define y data using f.e. computeLabels()")
        return self.y


    def getLabels(self):
        return self.get_y()


    def storeDfInCsvFile(self):
        """ Store df in csv file """
        # Make sure the directory exists
        os.makedirs(myconfig.DATA_DIR, exist_ok=True)
        # delete old train data file if True
        try:
            if self.overwrite:
                os.remove(self.csvFilePath)
        except Exception as e:
            print(str(e))
        # write df to new csv file and delete old content
        self.df.to_csv(self.csvFilePath, encoding='utf-8',
                       index=False,  # False: without index
                       sep=";")     # custom seperator
        # add problem name in first line
        with open(self.csvFilePath, "r+") as f:
            file_data = f.read()
            f.seek(0, 0)  # get the first line
            f.write(str(self.problem.name()) + '\n' + file_data)
        print(f"Successfully stored {self.problem.name()}-data to location:", self.csvFilePath)


    def plotNewData(self, title=""):
        """ Plot the generated data. Adds size N to title """
        # Extract the x and y data from the dataframe
        if len(self.df.columns) > 2:
            x1 = self.df[self.df.columns[0]]
            x2 = self.df[self.df.columns[1]]
        else:
            x1 = self.df[self.df.columns[0]]
        y = self.df[self.df.columns[-1]]

        # Create the plot
        if len(self.df.columns) > 2:
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
        title = title + f" (n={self.df.shape[0]})"
        plt.title(title)
        plt.show()


    def generateCsvFileWithNewInputX(self, plotData=False):
        """ KICKSTART: Generate random X data, compute labels and store as new CSV file """
        self.generateRandomX()
        self.computeLabels()
        if plotData:
            self.plotNewData(title=f"Newly generated data with mode='{self.mode}'")
        self.storeDfInCsvFile()


