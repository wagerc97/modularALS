"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Script with utility functions for mlmodel.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl

import sys
print("Python version:", sys.version)
assert sys.version_info >= (3,5)

print("Matplotlib version:", mpl._get_version())
#assert mpl.version?
import matplotlib.pyplot as plt

import sklearn
print("sklearn version:", sklearn.__version__)
assert sklearn.__version__ >= "0.20"

def mlSetup():
    # Matplot settings
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    return None

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    # Where to save the figures
    PROJECT_ROOT_DIR = ".."
    CHAPTER_ID = "training_linear_models"
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
    os.makedirs(IMAGES_PATH, exist_ok=True)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def defineNormalEquation():
    """ Defines a normal equation with n=100 """
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    print("X:\n", pd.DataFrame(X))
    print("y:\n", pd.DataFrame(y))
    return X, y


def plotNormalEquation(X, y):
    plt.figure(figsize=(5, 3))
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])  # crop plot window
    #save_fig("generated_data_plot")
    plt.show()
