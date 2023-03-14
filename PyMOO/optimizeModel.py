#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Wrapper to optimize a given ML model.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pymoo_helpers as helper
from optimizer import Optimizer
from problemWrapper import ProblemWrappedModel
from sklearn.metrics import r2_score
import myconfig as cfg

def main():

    ### invoke a new instance of Optimizer ###
    myOptimizer = Optimizer()

    ### Define Algorithm for optimization ###
    myOptimizer.setAlgorithm('NSGA2')       # genetic algorithm

    ### invoke a new instance of ProblemWrappedModel ###
    myProblem = ProblemWrappedModel()

    ### Load model pipeline from file ###
    #loadedModel = myOptimizer.fetchPipelineFromFile()
    loadedModel = ProblemWrappedModel.fetchPipelineFromFile()
    print("Loaded model:\n", loadedModel)



    #testScore = helper.getTestScore(loadedModel, X_test, y_test)
    #testScore = loadedModel.score(predData.X_test, predData.y_test)
    pred_y = loadedModel.predict(predictionInputData.X_test)
    testScore = r2_score(y_true=predictionInputData.y_test, y_pred=pred_y)
    print("Model test score (R²): ", round(testScore, 3))


if __name__ == '__main__':
    main()



