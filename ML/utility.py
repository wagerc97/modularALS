"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Script with utility functions for mlmodel.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import sys
import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def assertRequirements():
    print("Python version:", sys.version)
    assert sys.version_info >= (3, 5)
    #assert mpl.version?
    print("Matplotlib version:", mpl._get_version())
    print("sklearn version:", sklearn.__version__)
    assert sklearn.__version__ >= "0.20"

def mlSetup():
    # Matplot settings
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    print("-"*60, "\n")
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

def defineNormalEquation(verbose=False, n=100):
    """ Defines a normal equation with n=100 """
    print("Defining Normal Equation with n =", n)
    np.random.seed(42)  # Set the random seed to 42
    X = 2 * np.random.rand(n, 1)
    y = 4 + 3 * X + np.random.randn(n, 1)
    # convert dataframe
    X = pd.DataFrame(X, columns=["X"])
    y = pd.DataFrame(y, columns=["y"])
    if verbose:
        print("X:\n", X)
        print("y:\n", y)
    return X, y

def computeAnalyticalSolution(X, y, n=100):
    """
    Compute theta which minimizes the loss function.
    Note: Column-wise concatenation using np.c_
    :param n:
    :param X:
    :param y:
    :return:
    """
    # add x0 = 1 to each instance in a new column
    X_b = np.c_[np.ones((n, 1)), X]
    # Compute the (multiplicative) inverse of a matrix
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best

def plot1D(X, y, saveFileWithName=None):
    plt.figure(figsize=(9, 6))
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2,     # x-axis
              0, 15])   # y-axis
    if saveFileWithName:
        save_fig(saveFileWithName)
    plt.show()

def readDataFromCsvToDf(filepath=os.path.join("..", "data", "data.csv"), verbose=False):
    """
    Read in data training data from csv-file and store it in dataframe.
    :param filepath:
    :return: dataframe holding training data + problem name
    """
    # get data
    data = pd.read_csv(filepath, sep=';', header=1)
    # get problem name from first line
    import csv
    with open(filepath, "r", newline='\n') as f:
        reader = csv.reader(f)
        problem_name = next(reader)[0]
    print("Found data for problem:", problem_name)
    if verbose:
        print("csv filepath:", filepath)
        print(data)
    return data, problem_name

def plotData(df):
    # Extract the x and y data from the dataframe
    if 'x2' in df.columns:
        x1 = df['x1']
        x2 = df['x2']
        y = df['y']
    else:
        x1 = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    # Create the plot
    if 'x2' in df.columns:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, y)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
    else:
        fig, ax = plt.subplots()
        ax.plot(x1, y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.show()
    return None

