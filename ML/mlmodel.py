"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Definition of ML Model

Source: https://www.kaggle.com/code/wagerc97/aml-regression

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np  # maths and stuff
import pandas as pd  # data handling
import matplotlib.pyplot as plt  # plot stuff
import utility as util
from machineLearningModel import MachineLearningModel


if __name__ == '__main__':

    # Set up ML model environment
    util.mlSetup()

    # Define equation and get X, y
    X, y = util.defineNormalEquation(verbose=False)

    # Plot these input and output values
    # todo: some weird error with matplotlib
    util.plot1D(X, y, saveFileWithName="generated_data_plot")

    # Compute analytical solution
    theta_best = util.computeAnalyticalSolution(X,y)
    print("theta_best:\n", theta_best)

    # Split train data for k-fold cross validation
    # todo: for larger datasets

    # Create a model of choice
    myModel = MachineLearningModel('linear_regression', normalize=False)
    print(myModel)

    # Train model
    myModel.train(X_train=X, y_train=y)
    myModel.printTrainedParameters()

    # Use analytical solution theta to predict best result
    y_pred = myModel.predict(X_test=X, verbose=False)

    # Plot prediction
    myModel.plotPrediction(y_pred)

    # Get intercept and coefficients of trained model
    trainedParams = myModel.getTrainedParameters()

    # Assess deviation of prediction to true data
    # Question: Should we measure the accuracy or should we compare theta?
    #myModel.assessAccuracy(y_pred=y_pred)
    absoluteError = myModel.assessAccuracy2(theta_best=theta_best)
    print("absoluteError:",absoluteError)

