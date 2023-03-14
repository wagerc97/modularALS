#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Predict with saved ML model 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import ml_helpers as helper
from Data.dataHandler import DataHandler
from sklearn.metrics import r2_score
import myconfig as cfg

def main():

    ### invoke a new instance of DataHandler ###
    predictionInputData = DataHandler()

    ### provide data from csv file ###
    #todo: remove split for prediction
    predictionInputData.readDataFromCsvToDf(cfg.PRED_DATA_FILE)
    X_df, y_df = predictionInputData.splitDataForPrediction()

    # check dimension of splits
    print(f"\nX dimension: {X_df.shape} \n{X_df.head()}")
    print(f"\ny dimension: {y_df.shape} \n{y_df.head()}")

    ### Load model from file ###
    loadedModel = helper.loadModelFromFile()
    print("Loaded model:\n", loadedModel)

    #testScore = helper.getTestScore(loadedModel, X_test, y_test)
    #testScore = loadedModel.score(predData.X_test, predData.y_test)
    pred_y = loadedModel.predict(X_df)
    testScore = r2_score(y_true=y_df, y_pred=pred_y)
    print("Model test score (R²): ", round(testScore, 3))


if __name__ == '__main__':
    main()
