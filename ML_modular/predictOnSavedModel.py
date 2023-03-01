#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Predict with saved ML model 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt # dont delete this
import ml_helpers as helper
from machineLearningModel import MachineLearningModel
from dataHandler import DataHandler


def main():

    myDataHandler = DataHandler()

    ### Load model from file ###
    loadedModel = helper.loadModelFromFile()
    print("Loaded model:\n", loadedModel)

    ### Define TEST data ###
    myDataHandler.readDataFromCsvToDf(filename="new_data.csv")

    #testScore = helper.getTestScore(loadedModel, X_test, y_test)
    testScore = loadedModel.score(X_test, y_test)
    print("Model test score: ", round(testScore, 3))



if __name__ == '__main__':
    main()
