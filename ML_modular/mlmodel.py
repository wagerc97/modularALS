#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Wrapper to build several ml models with easy interface

Source: https://www.kaggle.com/code/wagerc97/aml-regression

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt # dont delete this
import ml_helpers as helper
import myconfig
from machineLearningModel import MachineLearningModel


def main():

    ### Check system requirements ###
    helper.assertRequirements()

    ### Set up M L model environment ###
    helper.mlSetup()

    ### Read data from csv to new df ###
    df, problem_name = helper.readDataFromCsvToDf(verbose=False)

    ### Plot data ###
    helper.plotRawData(df, title="original df")

    ### Create a model of choice ###
    myModel = MachineLearningModel()
    print(myModel)

    ### split data into train and test set ###
    myModel.defineDataSplits(df)
    X_train, y_train, X_test, y_test, train_df, _ = myModel.getDataSplits()
    print("\nX_train.head():\n", X_train.head())
    print("\ny_train.head():\n", y_test.head())

    ### Plot Train data ###
    #train_df = helper.concatenateDataframes(myModel.X_train, myModel.y_train)
    print("\nTrain df:\n", myModel.train_df)
    #helper.plotData(myModel.train_df, title="train data")

    ### Define score ###
    #score = helper.defineScore()
    myModel.defineScore("mae")

    ### Create Pipeline to easily configure estimator ###
    #pipeline = helper.createPipeline(normalize=True)
    myModel.createPipeline("krr", normalize=True)

    ### Define hyperparameters for grid search ###
    param_dict = {
        "kernelridge__alpha":[0.001, 0.01, 0.1, 1],
        "kernelridge__gamma": [0.001, 0.01, 0.03, 0.05, 0.1]
    }
    myModel.defineParamGrid(param_dict)

    ### Train model pipeline in Gridsearch with CV ###
    #krr_gscv, test_scores = helper.GridSearchCvForKrr(pipeline, score, X_train, y_train, X_test, y_test)
    myModel.applyGridSearchCV(verbose=True)
    print(f"Trained GridsearchCV object:\n{myModel.grid_search_cv}\n")

    ### Get best model ###
    bestModel = myModel.getBestEstimator()
    print(f"Best train score: {round(myModel.grid_search_cv.best_score_, 3)}") # Mean cross-validated score of the best_estimator
    print(f"Best parameters: {myModel.grid_search_cv.best_params_}") # Parameter setting that gave the best results on the hold out data.

    ### Get TEST accuracy of best model ###
    print("All CV TEST score:", myModel.getAllCvTestScores()) #>  [-90.58918323 -39.71723522 -98.90899151 -52.99895814 -85.60559008]
    print("Mean CV TEST score:", myModel.getMeanTestScoreOfGridsearchCV()) #> mean: -73.564
    print("TEST score of best model:", myModel.getTestScore()) #> 0.997

    ### Use analytical solution theta to predict best result ###
    y_pred = myModel.predict(X_test)

    ### Merge Dataframes ###
    #pred_Xtest_df = helper.concatenateDataframes(X_test, y_pred)

    ### Plot predicted points only ###
    #helper.plotData(pred_Xtest_df, title="Prediction on test data")

    ### Plot prediction against train data ####
    #helper.plotPredictionAndData(pred_df=pred_Xtest_df, train_df=train_df, title="Prediction vs train")
    myModel.plotPredictionAndTrainData(title="Prediction vs train")

    ### Store best model in external file ###
    #helper.saveModelToFile(bestModel)
    myModel.saveModelToFile()

    ### Load model from file ###
    #loadedModel, testScore = helper.loadModelFromFile(X_test, y_test)
    loadedModel = myModel.loadModelFromFile()
    print("Loaded model:\n", loadedModel)

    testScore = myModel.getTestScore(X_test)
    print("Model test score: ", round(testScore, 3))

    # Quellen:
    # Trivial: https://www.kaggle.com/code/wagerc97/uebung2-bsp2-angabe
    # KRR: https://www.kaggle.com/code/wagerc97/uebung1-bsp2-angabe


if __name__ == '__main__':
    main()
