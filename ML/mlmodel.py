#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Wrapper to build several ml models with easy interface

Source: https://www.kaggle.com/code/wagerc97/aml-regression

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
import matplotlib.pyplot as plt
import ml_helpers as helper
import myconfig


def main():

    ### Check system requirements ###
    #helper.assertRequirements()

    ### Set up ML model environment ###
    helper.mlSetup()

    ### Read data from csv to new df ###
    data, problem_name = helper.readDataFromCsvToDf(verbose=False)
    df = data.copy()    # copy data immediately

    ### Plot data ###
    helper.plotData(df, title="original df")

    ### split data into train and test set ###
    X_train, y_train, X_test, y_test = helper.splitData(df)
    print("\nX_train.head():\n", X_train.head())
    print("\ny_train.head():\n", y_train.head())

    ### Plot Train data ###
    train_df = helper.concatenateDataframes(X_train, y_train)
    print("\nTrain df:\n", train_df)    #> [80 rows x 3 columns]
    #helper.plotData(train_df, title="train data")

    ### Define score ###
    score = helper.defineScore()

    ### Create Pipeline to easily configure estimator ###
    pipeline = helper.createPipeline(normalize=True)

    ### Train model pipeline in Gridsearch with CV ###
    krr_gscv, test_scores = helper.GridSearchCvForKrr(pipeline, score, X_train, y_train, X_test, y_test)
    print(f"Trained GridsearchCV object:\n{krr_gscv}\n")

    ### Get best model ###
    bestModel = krr_gscv.best_estimator_
    print(f"Best train score: {round(krr_gscv.best_score_, 3)}") # Mean cross-validated score of the best_estimator
    print(f"Best parameters: {krr_gscv.best_params_}") # Parameter setting that gave the best results on the hold out data.

    ### Get TEST accuracy of best model ###
    print("Cross-validated TEST score of GridsearchCV:", test_scores) #>  [-90.58918323 -39.71723522 -98.90899151 -52.99895814 -85.60559008]
    print("Mean score of GridsearchCV:", round(np.mean(test_scores), 3)) #> mean: -73.564
    print("TEST score of best model:", round(bestModel.score(X_test, y_test), 3)) #> 0.997

    ### Use analytical solution theta to predict best result ###
    y_pred = bestModel.predict(X_test)

    ### Merge Dataframes ###
    pred_Xtest_df = helper.concatenateDataframes(X_test, y_pred)

    ### Plot predicted points only ###
    #helper.plotData(pred_Xtest_df, title="Prediction on test data")

    ### Plot prediction against train data ####
    helper.plotPredictionAndData(pred_df=pred_Xtest_df, train_df=train_df, title="Prediction vs train")

    ### Store best model in external file ###
    filename = "finalized_model.sav"
    filepath = os.path.join(myconfig.MODEL_DIR, filename)

    print("main:", filepath)
    helper.saveModelToFile(bestModel)

    ### Load model from file ###
    print("main:", filepath)
    loadedModel, testScore = helper.loadModelFromFile(filepath, X_test, y_test)

    print("Model test score: ", round(testScore, 3))
    print("Loaded model:\n", loadedModel)

    # Quellen:
    # Trivial: https://www.kaggle.com/code/wagerc97/uebung2-bsp2-angabe
    # KRR: https://www.kaggle.com/code/wagerc97/uebung1-bsp2-angabe


if __name__ == '__main__':
    main()
