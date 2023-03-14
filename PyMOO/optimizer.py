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

class Optimizer:
    """
    A class to optimize a given ml model and return the optimum.
    TODO: define mode (maximum or minimum)
    User can provide the algorithm upon initialization.
    """

    def __init__(self, algorithm_name='nsga2', seed=42):
        self.seed = seed                        # random seed
        self.pipeline = None                    # model pipeline
        self.algorithm_name = algorithm_name    # nsga2 is default
        self.algorithm = None                   # Pymoo algorithm object
        self.setAlgorithm()
        self.pop_size = None                    # population size in genetic algorithms
        self.iters = 100                        # number of iterations in minimization
        self.solution = None                    # optimized solution

        print(f"New {self.__class__.__name__} object created ")


    def __del__(self):
        """ destructor frees up memory """
        print(f"\nObject {self.__class__.__name__} destroyed")


    def setAlgorithm(self, param_algo=None):
        """ Create an algorithm. Default is 'NSGA2'. """
        if param_algo is None and self.algorithm_name is not None:
            param_algo = self.algorithm_name

        if param_algo.lower() == "nsga2" or "n":
            from pymoo.algorithms.moo.nsga2 import NSGA2
            # use algorithm specific default if no pop size was defined by user
            pop = 100 if self.pop_size is None else self.pop_size
            self.algorithm = NSGA2(pop_size=pop)
        else:
            raise Exception("Enter parameter { NSGA2 } ... i have not implemented any other algorithms yet")


    def setPopulationSize(self, param_pop_size):
        """ Set a new population size for genetic algorithms. Default depends on algorithm. """
        self.pop_size = param_pop_size


    def setIterations(self, param_iters):
        """ Set a new population size for genetic algorithms. Default depends on algorithm. """
        self.iters = param_iters


    def solve(self, verbose=False):
        """ Solve the problem with a chosen algorithm. """
        # Define Result
        # https://pymoo.org/interface/minimize.html
        iters = self.iters
        self.solution = minimize(
            problem=self.pipeline.predict(),
            algorithm=self.algorithm,
            ('n_gen', iters),   # n_gen defines the number of iterations
            seed=self.seed,
            verbose=verbose     # prints out solution in each iteration
            )
