#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

test the problem wrapper

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from problemWrappedModel import ProblemWrappedModel
import myconfig as cfg
import numpy as np
from pymoo.problems.functional import FunctionalProblem


def main():

    ### invoke a new instance of ProblemWrappedModel ###
    #myProblem = ProblemWrappedModel()

    ### Load model pipeline from file ###
    #loadedModel = myOptimizer.fetchPipelineFromFile()
    #loadedModel = ProblemWrappedModel.fetchPipelineFromFile()
    #print("Loaded model:\n", loadedModel)


    objs = [
        lambda x: np.sum((x - 2) ** 2),
        lambda x: np.sum((x + 2) ** 2)
    ]

    constr_ieq = [
        lambda x: np.sum((x - 1) ** 2)
    ]

    n_var = 10

    problem = FunctionalProblem(n_var,
                                objs,
                                constr_ieq=constr_ieq,
                                xl=np.array([-10, -5, -10]),
                                xu=np.array([10, 5, 10])
                                )

    F, G = problem.evaluate(np.random.rand(3, 10))

    print(f"F: {F}\n")
    print(f"G: {G}\n")



if __name__ == '__main__':
    main()

