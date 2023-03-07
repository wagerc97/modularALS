#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Compute and store X-Y value pairs from different problems. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pymoo_helpers as helper
from dataGenerator import DataGenerator

# Chose a problem
# - SOO: Rosenbrock (x1, x2, y)      (n_var: 2, n_obj: 1, n_constr: 0) -> solution is one single point
# - MOO: Zakharov   (x1, x2, y1, y2) (n_var: 2, n_obj: 2, n_constr: 0) -> solution is pareto front in 2D

# Some default values
SEED = 42
N = 123
PROBLEM = "Rosenbrock"
ALGORITHM = "nsga2"

def main(n=N, seed=SEED, problem_name=PROBLEM, algorithm=ALGORITHM, deleteOldData=True):


    ### Invoke a new instance of fresh data ###
    inputData = DataGenerator(n=n,
                              seed=seed,
                              problem_name=problem_name,
                              algorithm_name=algorithm,
                              deleteOldData=deleteOldData)

    ### Generate random X data, compute labels and store as new CSV file ###
    inputData.generateCsvFileWithNewInputX(plotData=True)

    """    
    ### Define Problem ###
    inputData.createProblem()

    ### Define random input values ###
    inputData.generateRandomX()

    ### Compute according output values ###
    inputData.computeLabels()

    ### Store df in csv file ###
    inputData.storeDfInCsvFile()

    ### Plot random values ###
    inputData.plotNewData(title="New data")
    """


if __name__ == '__main__':
    main()
