#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Predict with saved ML model 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import ml_helpers as helper
from data.dataHandler import DataHandler
from sklearn.metrics import r2_score
import myconfig as cfg

def main():

    ### invoke a new instance of DataHandler ###
    predictionInputData = DataHandler()

    ### provide data from csv file ###
    predictionInputData.readAndSplitFromFile(cfg.PRED_DATA_FILE)
    #todo: remove split for prediction

    ### Load model from file ###
    loadedModel = helper.loadModelFromFile()
    print("Loaded model:\n", loadedModel)

    #testScore = helper.getTestScore(loadedModel, X_test, y_test)
    #testScore = loadedModel.score(predData.X_test, predData.y_test)
    pred_y = loadedModel.predict(predictionInputData.X_test)
    testScore = r2_score(y_true=predictionInputData.y_test, y_pred=pred_y)
    print("Model test score (R²): ", round(testScore, 3))


if __name__ == '__main__':
    main()
