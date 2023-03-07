#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Wrapper to build several ml models with easy interface

Source: https://www.kaggle.com/code/wagerc97/aml-regression

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt # dont delete this
import pandas as pd
from sklearn.metrics import r2_score

import ml_helpers as helper
from modelGenerator import ModelGenerator


def main():

    ### Check system requirements ###
    helper.assertRequirements()

    ### Set up M L model environment ###
    #helper.mlSetup()

    ### Read data from csv to new df ###
    df, problem_name = helper.readDataFromCsvToDf(verbose=False)

    ### Plot data ###
    helper.plotRawData(df, title="original df")

    ### Create a model of choice ###
    myModel = ModelGenerator()
    print(myModel)

    ### split data into train and test set ###
    myModel.defineDataSplits(df)
    X_train, y_train, _, _, train_df, _ = myModel.getDataSplits()
    #print("\nX_train.head():\n", X_train.head())
    #print("\ny_train.head():\n", y_train.head())

    ### Plot Train data ###
    print("\nTrain df:\n", myModel.train_df)
    #helper.plotData(myModel.train_df, title="train data")

    ### Define score ###
    # [ mae, ... ]
    myModel.defineScorer("mae")

    ### Create Pipeline to easily configure estimator ###
    # [ krr, svr,... ]
    MODEL = "krr"
    myModel.createPipeline(MODEL, normalize=False)
    print("\nmodel pipeline:", myModel.getPipeline())

    ### Define hyperparameters for grid search ###
    if MODEL == "krr":
        param_dict_krr = {
            "alpha":[0.001, 0.01, 0.1, 1],
            "gamma": [0.05, 0.1, 0.5, 1, 5, 10],
            "kernel": ["rbf"]
        }
        PARAM_DICT = param_dict_krr
    elif MODEL == "SVR":
        param_dict_svr = {
            "gamma": [0.1, 0.5, 1, 5, 10, 50],               # kernel coefficients
            "epsilon": [0.001, 0.01, 0.05, 0.1, 0.5, 1],     # epsilon tube
            "C": [1000, 5000, 10000, 50000, 100000, 500000],    # regularization, C=1/2alpha
            "kernel": ["rbf"]
        }
        PARAM_DICT = param_dict_svr
    else:
        raise Exception(f"Model type not given in mlmodel.py (given type was {MODEL})")

    myModel.defineParamGrid(PARAM_DICT)

    ### Train model pipeline in Gridsearch with CV ###
    # verbose will print results-table
    myModel.applyGridSearchCV(verbose=True)

    ### Get best model ###
    print(f"Best train score: {round(myModel.grid_search_cv.best_score_, 3)}") # Mean cross-validated score of the best_estimator during training
    print(f"Best parameters: {myModel.grid_search_cv.best_params_}") # Parameter setting that gave the best results on the hold out data.

    ### Get TEST accuracy of best model ###
    print("All CV TEST score:", myModel.getAllCvTestScores()) #>  [-90.58918323 -39.71723522 -98.90899151 -52.99895814 -85.60559008]
    print("Mean CV TEST score:", myModel.getMeanTestScoreOfGridsearchCV()) #> mean: -73.564
    print("TEST score of best model:", myModel.getTestScore()) #> 0.997

    ### Plot prediction against train data ####
    myModel.predict(X_test=myModel.X_test)
    myModel.plotPredictionAndTrainData(title="Prediction vs train")

    ### Save best model in external file ###
    myModel.saveModelToFile()

    ### Save train results in file ###
    myModel.saveDfToTemporaryFile()

    ### Load model from file ###
    loadedModel = myModel.loadModelFromFile()
    print("Loaded model:\n", loadedModel)

    ### Compute model score ###
    testScoreFromFunction = myModel.getTestScore(myModel.X_test)
    print("Model test score from function (R²): ", round(testScoreFromFunction, 3))

    """ Here, I compared the score functions """
    #best_estimator = myModel.getBestEstimator()
    #pred_y = best_estimator.predict(myModel.X_test)
    #pred_y = pd.DataFrame(pred_y, columns=["y"])
    #print("\n\n")
    #print(myModel.y_test)
    #print("\n")
    #print(pred_y)
    #testScore = r2_score(y_true=myModel.y_test, y_pred=pred_y)
    #print("Model test score from r2_score (R²): ", round(testScore, 3))



    # Quellen:
    # SVR: https://www.kaggle.com/code/wagerc97/uebung2-bsp2-angabe
    # KRR: https://www.kaggle.com/code/wagerc97/uebung1-bsp2-angabe


if __name__ == '__main__':
    main()
