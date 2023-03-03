#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Predict with saved ML model 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import ml_helpers as helper
from data.dataHandler import DataHandler


def main():

    ### invoke a new instance of DataHandler ###
    dataHandler = DataHandler()

    ### provide data from csv file ###
    dataHandler.kickStartFromFile()

    ### Load model from file ###
    loadedModel = helper.loadModelFromFile()
    print("Loaded model:\n", loadedModel)

    #testScore = helper.getTestScore(loadedModel, X_test, y_test)
    testScore = loadedModel.score(dataHandler.X_test, dataHandler.y_test)
    print("Model test score: ", round(testScore, 3))


if __name__ == '__main__':
    main()
