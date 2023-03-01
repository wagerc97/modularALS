#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Script with utility functions for MachineLearning module
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import sys
import csv
import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
import myconfig


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
    print("\n")
    print("+" * 30)
    print(f"{__name__} is here")
    print("+" * 30, "\n")
    return None


### NORMAL EQUATION FUNCTIONS ###
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

#################################

def readDataFromCsvToDf(filepath=myconfig.DATA_FILE, verbose=False):
    """ Read in data training data from csv-file and save it in dataframe. """
    # get data
    data = pd.read_csv(filepath, sep=';', header=1)
    # get problem name from first line
    with open(filepath, "r", newline='\n') as f:
        reader = csv.reader(f)
        problem_name = next(reader)[0]
    print("Found data for problem:", problem_name)
    if verbose:
        print("csv filepath:", filepath)
        print(data)
    return data, problem_name


def plotRawData(df, title=""):
    # Extract the x and y data from the dataframe
    if len(df.columns) > 2:
        x1 = df[df.columns[0]]
        x2 = df[df.columns[1]]
    else:
        #x1 = df.iloc[:, :-1]
        x1 = df[df.columns[0]]
    y = df[df.columns[-1]]

    # Create the plot
    if len(df.columns) > 2:
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
    title = title+f" (n={df.shape[0]})"
    plt.title(title)
    plt.show()
    return None

def printSimplifiedTable(ptable):
    print("\n+--------------------------------------------------------+")
    pd.set_option('display.max_columns', None) # print all columns
    print(ptable.drop(ptable.filter(regex='time|split|params|std_|param_').columns, axis=1).sort_values("rank_test_score").head())
    print("+--------------------------------------------------------+\n")
    return None


def concatenateDataframes(x_df, y):
    """ Merge y to end of Dataframes """
    combined_df = x_df.assign(y=y)
    return combined_df


def plotPredictionAndData(pred_df, train_df, title):
    # Check dimensionality of data
    if len(train_df.columns) > 2: # 2D
        # define figure shape
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for df, color, label in (pred_df, "r", "Prediction"+f" (n={pred_df.shape[0]})"), (train_df, "C0", "Train"+f" (n={train_df.shape[0]})"):
            # Extract the x and y data from the dataframe
            x1 = df[df.columns[0]]
            x2 = df[df.columns[1]]
            y = df[df.columns[-1]]
            # plot x,y
            ax.scatter(x1, x2, y, color=color, label=label)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('y')

    else: # 1D
        # define figure shape
        fig, ax = plt.subplots()
        for df, color, label in (pred_df, "r", "Prediction"+f" (n={pred_df.shape[0]})"), (train_df, "C0", "Train"+f" (n={train_df.shape[0]})"):
            # Extract the x and y data from the dataframe
            x1 = df[df.columns[0]]
            y = df[df.columns[-1]]
            # plot x,y
            ax.plot(x1, y, color=color)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
    title = title+f" (n={pred_df.shape[0]+train_df.shape[0]})"
    plt.title(title)
    ax.legend()
    plt.show()
    return None


def saveModelToFile(model, filepath=myconfig.MODEL_FILE):
    """save the model to disk"""
    # make sure the file destination exists
    os.makedirs(myconfig.MODEL_DIR, exist_ok=True)  # Make sure the directory exists
    joblib.dump(model, filepath)
    print("\nModel saved to file:", filepath)
    return None


def loadModelFromFile(filepath=myconfig.MODEL_FILE):
    """load the model from disk"""
    print("\nLoad model from file:", filepath)
    loadedModel = joblib.load(filepath)
    return loadedModel


def getTestScore(model, X_test, y_test):
    """ compute score of model on test data """
    return round(model.score(X_test, y_test), 3)


