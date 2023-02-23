"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Wrapper to build several ml models with easy interface

Source: https://www.kaggle.com/code/wagerc97/aml-regression

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np  # maths and stuff
import pandas as pd  # data handling
import matplotlib.pyplot as plt  # plot stuff
import utility as util
from machineLearningModel import MachineLearningModel


if __name__ == '__main__':
    # Check system requirements
    #util.assertRequirements()

    # Set up ML model environment
    util.mlSetup()

    # Read data from csv to new df
    data, problem_name = util.readDataFromCsvToDf(verbose=False)
    df = data.copy()    # copy data immediately

    #X = df.iloc[:, :(len(df.columns)-1)]    # get every column except the last
    #print("X:\n", X)
    #y = df.iloc[:, :-1]  # get last column
    #print("y:\n", y)

    # Plot data
    util.plotData(df)

    # split data into train and test set
    X_train, y_train, X_test, y_test = util.splitData(df)
    print("\nX_train.head():\n", X_train.head())
    print("\ny_train.head():\n", y_train.head())

    # In sklearn ist ein höherer Score immer besser. Der mean_absolute_error (MAE) ist aber besser, je kleiner er ist.
    # Wenn wir den Scorer erstellen nehmen wir also - mean_absoute_error als Bewertungsmaß. Dazu setzten wir greater_is_better=False.
    # Dementsprechend werden die Scores im Grid Search auch negativ sein und der Score, der am nähesten zu 0 ist der beste.
    from sklearn.metrics import mean_absolute_error, make_scorer
    score = make_scorer(mean_absolute_error, greater_is_better=False)


    # Pipeline to easily configure estimator
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.kernel_ridge import KernelRidge
    pipeline_krr = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", KernelRidge(kernel="rbf"))
    ])

    # Get best model
    bestModel = util.GridSearchCvForKrr(pipeline_krr, score, X_train, y_train)
    print(bestModel)

    # Use analytical solution theta to predict best result
    #y_pred = myModel.predict(X_test=X_test, verbose=False)
    y_pred = bestModel.predict(X_test)
    print(X_test.shape)
    print(y_pred.shape)

    # Merge Dataframes
    pred_Xtest_df = util.concatenateDataframes(X_test, y_pred)
    print("\nCombined df:\n", pred_Xtest_df)

    # Plot prediction
    #myModel.plotPrediction(y_pred)
    #util.plotData(pred_Xtrain_df)
    util.plotData(pred_Xtest_df)

    # Quelle: https://www.kaggle.com/code/wagerc97/uebung2-bsp2-angabe


    # Get intercept and coefficients of trained model
    #trainedParams = myModel.getTrainedParameters()

    # Compute analytical solution
    #...

    # Assess deviation of prediction to true data
    # Question: Should we measure the accuracy or should we compare theta?
    #myModel.assessAccuracy(y_pred=y_pred)
    #absoluteError = myModel.assessAccuracy2(theta_best=theta_best)
    #print("absoluteError:",absoluteError)

