#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

test the problem wrapper

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from problemWrappedModel import ProblemWrappedModel
import myconfig as cfg
import numpy as np
from pymoo.problems.functional import FunctionalProblem
from dataGenerator import DataGenerator
import pymoo_helpers as helper

def main():
    #objs = [
    #    lambda x: np.sum((x - 2) ** 2),
    #    lambda x: np.sum((x + 2) ** 2)
    #]
    #constr_ieq = [
    #    lambda x: np.sum((x - 1) ** 2)
    #]

    ### DATA ###

    #inputX = np.random.rand(3, n_var)
    n = 15
    datGen = DataGenerator(mode="predict", n=n)
    datGen.generateRandomX()
    inputX = datGen.get_X()

    # Print shape and dataframe if small enough
    print("\n"); print("-"*30)
    print(f"inputX shape = {inputX.shape} \n{inputX if (inputX.shape[0] * inputX.shape[1]) < 25 else 'dimensions too high to print' }")
    print("-"*30, "\n")

    # Using the same problem for problem wrapper arguments as in DataGenerator

    problem = datGen.getProblem()
    n_var, n_obj, n_ieq_constr, xl, xu = datGen.getProblemParameters()

    # Print Problem parameter
    #for arg in n_var, n_obj, n_ieq_constr, xl, xu:
    #    print(arg)

    ### PROBLEM ###

    """
    n_var = 2
    problemWrapper = ProblemWrappedModel(
        n_var=n_var,  # hardcoded, but problem dependent
        n_obj=1,  # hardcoded, but problem dependent
        # n_ieq_constr=len(constr_ieq),
        n_ieq_constr=1,
        xl=-2, xu=2  # hardcoded, but problem dependent
    )
    """
    problemWrapper = ProblemWrappedModel(
        n_var=n_var,                    # hardcoded, but problem dependent
        n_obj=n_obj,                    # hardcoded, but problem dependent
        n_ieq_constr=n_ieq_constr+1,    # todo: was ist n_ieq_constr??
        xl=xl, xu=xu                    # hardcoded, but problem dependent
    )

    # provide ml model for problem wrapper
    problemWrapper.fetchPipelineFromFile()
    print("loaded model:", problemWrapper.pipeline)


    # Compute labels i.e. evaluate ml model wrapped in pymoo problem at random inputX positions
    problemWrapper.computeLabels(X=inputX)
    result = problemWrapper.getResult()

    #print("result", result)

#####################################################################################################

    datGen.computeLabels()
    #labels = datGen.get_y()
    labels = datGen.getLabels()

    #print("\nlabels:", labels)

#####################################################################################################

    combined_df = helper.concatenateDataframes(result, labels)
    print("Final result:\n", combined_df)


    del datGen
    del problemWrapper

if __name__ == '__main__':
    main()
