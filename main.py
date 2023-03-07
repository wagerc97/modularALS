#!/usr/bin/python3

from myconfig import *
import sys
# Bring your packages onto the path
sys.path.append(os.path.abspath(os.path.join('.', 'ML')))  # '.' for main in .../Optimization/main.py
sys.path.append(os.path.abspath(os.path.join('.', 'PyMOO')))

# Settings for procedure
SEED = 42
N = 60
PROBLEM = "Rosenbrock"
ALGORITHM = "nsga2"
overWriteOldCsvData = True

if __name__ == '__main__':

    ### Create Train Data ###
    print("\n\n=====================================================================\n")
    print(">>> Generate new train data <<<\n")
    from PyMOO.dataGenerator import DataGenerator
    inputDataGenerator = DataGenerator(mode="train",
                                       n=N,
                                       seed=SEED,
                                       problem_name=PROBLEM,
                                       algorithm_name=ALGORITHM,
                                       overwrite=overWriteOldCsvData)

    ### Generate random X data, compute labels and store as new CSV file ###
    inputDataGenerator.generateCsvFileWithNewInputX(plotData=True)


    ### Create, train, evaluate and store ml model ###
    print("\n\n=====================================================================\n")
    print(">>> Generate and train ML model <<<\n")
    import ML.generateModel as gml
    gml.main()


    ### Create Prediction Data ###
    print("\n\n=====================================================================\n")
    print(">>> Generate new unseen data to predict on <<<\n")
    from PyMOO.dataGenerator import DataGenerator
    predDataGenerator = DataGenerator(mode="predict")

    ### Generate input data X ###
    predDataGenerator.generateCsvFileWithNewInputX(plotData=True)


    ### Predict with saved model ###
    print("\n\n=====================================================================\n")
    print(">>> Predict on new data <<<\n")
    import ML.predictOnSavedModel as psml
    psml.main()

