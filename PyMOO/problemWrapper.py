#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This class wraps an ml model as pymoo problem. This allows the pymoo minimize function to optimize this 
wrapped problem. 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import joblib

import myconfig
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.problems.functional import FunctionalProblem

# Ignore UserWarning
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


# Link: https://pymoo.org/problems/definition.html#nb-problem-definition-functional

class ProblemWrapper(Problem):
    """ Wraps the ML model in a PyMOO Problem class """

    def __init__(self, n_var=None, n_obj=None, n_ieq_constr=None, xl=None, xu=None):

        # initialisation of parent class requires certain parameters
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)

        # Attributes
        self.df = None
        self.result = None
        
        self.pipeline = None
        print(f"New {self.__class__.__name__} object created ")


    def __del__(self):
        """ destructor frees up memory """
        print(f"---Object {self.__class__.__name__} destroyed")



    def _evaluate(self, x, out, *args, **kwargs):
        """ Link: https://pymoo.org/interface/result.html
        res.X: Design space values are
        res.F: Objective spaces values
        res.G: Constraint values

        :param x: input array
        :param out: ???
        """
        #out["F"] = np.sum((x - 0.5) ** 2, axis=1)
        #out["G"] = 0.1 - out["F"]

        out["F"] = self.pipeline.predict(x)
        #out["G"] = 0.1 - out["F"]
        out["G"] = [0]  #todo: how to omit the calculation of this dictionary entry?


    def computeLabels(self, X):
        """ Use pymoo problem to compute y for labels. """
        # Apply the evaluate_df_row function to each row of the dataframe
        results = np.apply_along_axis(self.evaluate, axis=1, arr=X)
        #print("results in computeLabels:\n", results)

        results = results[:, 0]  # effectively drop the second inner entry
        results = results.reshape((-1,))  # reshape the array into a 1-dimensional array
        #print("results in computeLabels:\n", results)

        self.result = pd.DataFrame(results, columns=["y_wrap"])
        # Merge and store the two dataframes as one to use with ml model
        self.concatenateDataframes(X, self.result)


    def concatenateDataframes(self, x, y):
        """ Merge Dataframes and store in new dataframe to use with ml model """
        x_df = pd.DataFrame(x)
        frames = [x_df, y]
        combined_df = pd.concat(frames, axis=1)
        self.df = combined_df


    def getResult(self):
        if self.result is None:
            raise ValueError(f"Instance of {self.__class__.__name__} has no result yet. Please compute result first using computeLabels()")
        return self.result


    def fetchPipelineFromFile(self, filepath=myconfig.MODEL_FILE):
        """ Load the model from disk with provided path. """
        print("\nLoad model from file:", filepath)
        self.pipeline = joblib.load(filepath)


    def setPipelineObject(self, pipeline_object):
        """ Provide pipeline object as input parameter and save it in class. """
        self.pipeline = pipeline_object




