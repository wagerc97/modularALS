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

    X = df.iloc[:, :(len(df.columns)-1)]    # get every column except the last
    print("X:\n", X)
    y = df.iloc[:, :-1]  # get last column
    print("y:\n", y)

    # Plot data
    util.plotData(df)

    # Create a model of choice
    myModel = MachineLearningModel('linear_regression', normalize=False)
    print(myModel)

    # Train model
    myModel.train(X_train=X, y_train=y)

    # Print trained parameters
    myModel.printTrainedParameters()

    # Use analytical solution theta to predict best result
    y_pred = myModel.predict(X_test=X, verbose=False)

    # Plot prediction
    myModel.plotPrediction(y_pred)

    # Get intercept and coefficients of trained model
    trainedParams = myModel.getTrainedParameters()

    # Compute analytical solution
    #...

    # Assess deviation of prediction to true data
    # Question: Should we measure the accuracy or should we compare theta?
    #myModel.assessAccuracy(y_pred=y_pred)
    absoluteError = myModel.assessAccuracy2(theta_best=theta_best)
    print("absoluteError:",absoluteError)

