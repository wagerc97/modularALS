"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Wrapper to build several ml models with easy interface

Source: https://www.kaggle.com/code/wagerc97/aml-regression

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utility as util
from machineLearningModel import MachineLearningModel

if __name__ == '__main__':
    ### Check system requirements ###
    #util.assertRequirements()

    ### Set up ML model environment ###
    util.mlSetup()

    ### Read data from csv to new df ###
    data, problem_name = util.readDataFromCsvToDf(verbose=False)
    df = data.copy()    # copy data immediately

    ### Plot data ###
    util.plotData(df, title="original df")

    ### split data into train and test set ###
    X_train, y_train, X_test, y_test = util.splitData(df)
    print("\nX_train.head():\n", X_train.head())
    print("\ny_train.head():\n", y_train.head())

    ### Plot Train data ###
    train_df = util.concatenateDataframes(X_train, y_train)
    print("\nTrain df:\n", train_df)    #> [80 rows x 3 columns]
    #util.plotData(train_df, title="train data")

    ### Define score ###
    score = util.defineScore()

    ### Create Pipeline to easily configure estimator ###
    pipeline_krr = util.createPipeline(normalize=True)

    ### Get best model ###
    bestModel, scores = util.GridSearchCvForKrr(pipeline_krr, score, X_train, y_train, X_test, y_test)
    print(bestModel)

    ### Use analytical solution theta to predict best result ###
    y_pred = bestModel.predict(X_test)
    #print(f"y_pred: len={len(y_pred)}\n", y_pred) #> len=20, no NAN
    #print(f"X_test: len={len(X_test)}\n", X_test) #> len=20, no NAN

    ### Merge Dataframes ###
    pred_Xtest_df = util.concatenateDataframes(X_test, y_pred)
    #print("\nCombined df:\n", pred_Xtest_df)
    #print(pred_Xtest_df.shape) #> (35, 3)

    ### Plot prediction ###
    util.plotData(pred_Xtest_df, title="Prediction on test data")

    ### Plot prediction against train data ####
    util.plotPredictionAndData(pred_df=pred_Xtest_df, train_df=train_df, title="Prediction vs train")

    ### Get Accuracy of model ###
    #TODO: ich habe GridSearchCV -> brauche ich dann noch extra eine cross_validate??
    #TODO: Shoul i use a different score?

    print("\npipeline.score:", round(bestModel.score(X_test, y_test), 3)) #> -26.773
    print("\ncross_val_scores:\n", scores)
    #>  [-90.58918323 -39.71723522 -98.90899151 -52.99895814 -85.60559008]
    print("\nmean:", round(np.mean(scores),3)) #> mean: -73.564

    # Quelle: https://www.kaggle.com/code/wagerc97/uebung2-bsp2-angabe


