#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Wrapper to build several ml models with easy interface

Source: https://www.kaggle.com/code/wagerc97/aml-regression

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import matplotlib.pyplot as plt # dont delete this
import ml_helpers as helper
from machineLearningModel import MachineLearningModel


def main():

    myModel = MachineLearningModel()

    ### Load model from file ###
    loadedModel = myModel.loadModelFromFile()
    print("Loaded model:\n", loadedModel)

    testScore = loadedModel.getTestScore()
    #> AttributeError: 'Pipeline' object has no attribute 'getTestScore'

    print("Model test score: ", round(testScore, 3))




if __name__ == '__main__':
    main()
