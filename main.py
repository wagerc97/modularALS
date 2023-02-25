#!/usr/bin/python3


from myconfig import *
# Bring your packages onto the path
import os, sys
sys.path.append(os.path.abspath(os.path.join('.', 'ML')))  # '.' for main in .../Optimization/main.py
sys.path.append(os.path.abspath(os.path.join('.', 'PyMOO')))


problem = "Rosenbrock"
seed = 42
N = 100

if __name__ == '__main__':

    testiamhere()

    ### Create Train Data ###
    import PyMOO.getTrainData as gtd
    gtd.main(n=N, deleteOldData=True)


    ### Create, train, evaluate and store ml model ###
    import ML.mlmodel as mlm
    mlm.main()

