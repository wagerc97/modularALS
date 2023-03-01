#!/usr/bin/python3


from myconfig import *
# Bring your packages onto the path
import os, sys
sys.path.append(os.path.abspath(os.path.join('.', 'ML_modular')))  # '.' for main in .../Optimization/main.py
sys.path.append(os.path.abspath(os.path.join('.', 'PyMOO')))


problem = "Rosenbrock"
seed = 42
N = 300

if __name__ == '__main__':

    testiamhere()

    ### Create Train Data ###
    import PyMOO.getTrainData as gtd
    gtd.main(n=N, deleteOldData=True)

    ### Create, train, evaluate and store ml model ###
    import ML_modular.generateModel as gml
    gml.main()

    ### Predict with saved model ###
    #import ML_modular.predictOnSavedModel as psml
    #psml.main()

