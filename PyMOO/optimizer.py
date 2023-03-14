#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Run optimization algorithm over ML model to find optimal solution.
The model approximates the original mathematical problem. 
Thus, the optimal solution only approximates the analytical solution.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import joblib
import myconfig

import pymoo_helpers as helper
import os
import pandas as pd  # data handling
import numpy as np  # maths and stuff
import matplotlib.pyplot as plt  # plot stuff
from pymoo.optimize import minimize  # minimize solution
from pymoo.algorithms.moo.nsga2 import NSGA2


class Optimizer:
    """
    A class to optimize a given ml model and return the optimum.
    TODO: define mode (maximum or minimum)
    User can provide the algorithm upon initialization.
    """

    def __init__(self, algorithm_name='nsga2', seed=42):
        self.problem = None                     # pymoo problem object
        self.seed = seed                        # random seed
        self.algorithm_name = algorithm_name    # nsga2 is default
        self.algorithm = None                   # Pymoo algorithm object
        self.pop_size = None                    # population size in genetic algorithms
        self.iters = 100                        # number of iterations in minimization
        self.solution = None                    # optimized solution

        # Validation
        self.setAlgorithm()
        print(f"New {self.__class__.__name__} object created ")


    def __del__(self):
        """ destructor frees up memory """
        print(f"---Object {self.__class__.__name__} destroyed")


    def setAlgorithm(self, param_algo=None):
        """ Create an algorithm. Default is 'NSGA2'. """
        if param_algo is None and self.algorithm_name is not None:
            param_algo = self.algorithm_name

        if param_algo.lower() == "nsga2" or "n":
            # use algorithm specific default if no pop size was defined by user
            n_pop = 100 if self.pop_size is None else self.pop_size
            self.algorithm = NSGA2(pop_size=n_pop)
        else:
            raise Exception("Enter parameter { NSGA2 } ... i have not implemented any other algorithms yet")


    def setPopulationSize(self, pop_size):
        """ Set a new population size for genetic algorithms. Default depends on algorithm. """
        self.pop_size = pop_size


    def setIterations(self, iterations):
        """ Set a new population size for genetic algorithms. Default depends on algorithm. """
        self.iters = iterations


    def setProblem(self, problem):
        """ provide a pymoo problem object """
        self.problem = problem


    def solve(self, verbose=False):
        """ Solve the problem with a chosen algorithm. """
        if self.problem is None:
            raise ValueError("Provide a problem to optimizer")

        self.solution = minimize(problem=self.problem,
                                 algorithm=self.algorithm,   # n_gen defines the number of iterations
                                 termination=('n_gen', self.iters),
                                 seed=self.seed,
                                 verbose=verbose             # prints out solution in each iteration
                                 )
