#!/usr/bin/python3

from myconfig import *
import sys
# Bring your packages onto the path
sys.path.append(os.path.abspath(os.path.join('.', 'ML_modular')))  # '.' for main in .../Optimization/main.py
sys.path.append(os.path.abspath(os.path.join('.', 'PyMOO')))

# Settings for procedure
SEED = 42
N = 66
PROBLEM = "Rosenbrock"
ALGORITHM = "nsga2"
overWriteOldCsv = True

if __name__ == '__main__':

    testiamhere()

    ### Create Train Data ###
    from PyMOO.freshData import FreshData
    inputData = FreshData(n=N,
                          seed=SEED,
                          problem_name=PROBLEM,
                          algorithm_name=ALGORITHM,
                          deleteOldData=overWriteOldCsv)

    ### Generate random X data, compute labels and store as new CSV file ###
    inputData.kickStartCsvFileGeneration(plotData=True)


    ### Create, train, evaluate and store ml model ###
    import ML_modular.generateModel as gml
    gml.main()

    ### Predict with saved model ###
    #import ML_modular.predictOnSavedModel as psml
    #psml.main()

