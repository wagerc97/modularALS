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
    pipeline = util.createPipeline(normalize=True)

    ### Train model pipeline in Gridsearch with CV ###
    krr_gscv, test_scores = util.GridSearchCvForKrr(pipeline, score, X_train, y_train, X_test, y_test)
    print(f"Trained GridsearchCV object:\n{krr_gscv}\n")

    ### Get best model ###
    bestModel = krr_gscv.best_estimator_
    print(f"Best train score: {krr_gscv.best_score_}") # Mean cross-validated score of the best_estimator
    print(f"Best parameters: {krr_gscv.best_params_}") # Parameter setting that gave the best results on the hold out data.

    ### Get TEST accuracy of best model ###
    print("TEST score of best model:", round(bestModel.score(X_test, y_test), 3)) #> 0.997
    print("Cross-validated TEST score of GridsearchCV:", test_scores) #>  [-90.58918323 -39.71723522 -98.90899151 -52.99895814 -85.60559008]
    print("Mean score of GridsearchCV:", round(np.mean(test_scores), 3)) #> mean: -73.564

    ### Use analytical solution theta to predict best result ###
    y_pred = bestModel.predict(X_test)

    ### Merge Dataframes ###
    pred_Xtest_df = util.concatenateDataframes(X_test, y_pred)

    ### Plot predicted points only ###
    #util.plotData(pred_Xtest_df, title="Prediction on test data")

    ### Plot prediction against train data ####
    util.plotPredictionAndData(pred_df=pred_Xtest_df, train_df=train_df, title="Prediction vs train")

    ### Store best model in external file ###
    dirname = "models"
    filename = "finalized_model.sav"
    savepath = os.path.join(".", dirname, filename)
    util.saveModelToFile(filepath=savepath, model=bestModel)

    ### Load model from file ###
    loadedModel, testScore = util.loadModelFromFile(savepath, X_test, y_test)
    print("Model test score: ", round(testScore, 3))
    print("Loaded model:\n", loadedModel)

    # Quellen:
    # Trivial: https://www.kaggle.com/code/wagerc97/uebung2-bsp2-angabe
    # KRR: https://www.kaggle.com/code/wagerc97/uebung1-bsp2-angabe


    #TODO: ich habe GridSearchCV -> brauche ich dann noch extra eine cross_validate??
    #TODO: Should i use a different score?
