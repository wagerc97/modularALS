"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Definition of ML Model

Source: https://www.kaggle.com/code/wagerc97/aml-regression

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import utility as util
import os
import numpy as np  # maths and stuff
import pandas as pd  # data handling
import matplotlib.pyplot as plt  # plot stuff

if __name__ == '__main__':
    util.mlSetup()

    # Define equation and get X, y
    X, y = util.defineNormalEquation()

    # Plot these input and output values
    util.plotNormalEquation(X,y)


