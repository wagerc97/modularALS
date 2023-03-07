#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Predict with saved ML model 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import ml_helpers as helper
from data.dataHandler import DataHandler
import myconfig as cfg

def main():

    ### invoke a new instance of DataHandler ###
    predData = DataHandler()

    ### provide data from csv file ###
    predData.readAndSplitFromFile(cfg.PRED_DATA_FILE)

    ### Load model from file ###
    loadedModel = helper.loadModelFromFile()
    print("Loaded model:\n", loadedModel)

    #testScore = helper.getTestScore(loadedModel, X_test, y_test)
    testScore = loadedModel.score(predData.X_test, predData.y_test)
    print("Model test score: ", round(testScore, 3))


if __name__ == '__main__':
    main()
