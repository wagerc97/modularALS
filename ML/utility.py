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

def plotData(df, title=""):
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
    plt.title(title)
    plt.show()
    return None


def simplifyTable(ptable):
    print("\nNumber of columns in table:", len(ptable))
    pd.set_option('display.max_columns', None)
    return ptable.drop(ptable.filter(regex='time|split|params|std_|param_').columns, axis=1).sort_values("rank_test_score").head()


#TOdo: to model class
def GridSearchCvForKrr(pipeline_krr, score, X_train, y_train, X_test, y_test):
    from sklearn.model_selection import GridSearchCV
    # https://www.kaggle.com/code/wagerc97/uebung1-bsp2-angabe

    # Exhaustively search for best hyperparameters with GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline_krr,  # fresh estimator
        param_grid=[{  # hyperparameters
            "ridge__alpha": [0.001, 0.01, 0.1, 1],  # regularization strength, reduces variance of estimates, alpha=1/2C
            "ridge__gamma": [0.001, 0.01, 0.03, 0.05, 0.1],  #
            "ridge__kernel": ["rbf"]
        }],
        n_jobs=-1,      # jobs to run in parallel (-1 uses all processes available)
        scoring=score,  # using a callable to score each model
        cv=5,           # k-fold cross validation
        verbose=1
    )

    # fit the newly established model with data
    trained_gs = grid_search.fit(X_train, y_train)

    #TODO: cross-validation after gridsearchCV good?
    from sklearn.model_selection import cross_val_score
    test_scores = cross_val_score(trained_gs, X_test, y_test)

    # table of k cross validation results
    table_krr = pd.DataFrame(grid_search.cv_results_)
    #table_krr.sort_values("rank_test_score").head()
    print(simplifyTable(table_krr))

    return grid_search, test_scores


def splitData(df):
    from sklearn.model_selection import train_test_split

    # Data split
    data_train, data_test = train_test_split(df.copy(), test_size=0.2, random_state=42)
    # assign X all columns except "y" (which we want to predict)
    # assign y the "y" column
    X_train, y_train = data_train.drop(["y"], axis=1), data_train.y
    X_test, y_test = data_test.drop(["y"], axis=1), data_test.y
    return X_train, y_train, X_test, y_test


def concatenateDataframes(x_df, y):
    """ Merge y to end of Dataframes """
    #y_df = pd.DataFrame(y, columns=["y"])  # predict() only returns numpy.ndarray
    #frames = [x_df, y_df]

    #combined_df = pd.concat(frames, axis=1, ignore_index=True)
    #combined_df = pd.concat(frames, axis=1)
    #combined_df = x_df+y_df
    #combined_df = pd.merge(x_df, y_df)
    combined_df = x_df.assign(y=y)

    return combined_df


#TODO: move to model class
def createPipeline(normalize=False):
    # Create Pipeline to easily configure estimator
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.kernel_ridge import KernelRidge

    #TODO: create pipeline modular
    # empty pipeline
    # define model
    #model = ("ridge", KernelRidge(kernel="rbf"))
    #pipeline.append(model)
    #pipeline = [model]
    # define normalization method
    #if normalize:
    #    normalizer = ("scale", StandardScaler())
    #    pipeline.append(normalizer)
    # Create pipeline object
    #pipeline_krr = Pipeline(pipeline)

    pipeline_krr = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", KernelRidge(kernel="rbf"))
    ])
    return pipeline_krr


def defineScore():
    # In sklearn ist ein höherer Score immer besser. Der mean_absolute_error (MAE) ist aber besser, je kleiner er ist.
    # Wenn wir den Scorer erstellen nehmen wir also - mean_absoute_error als Bewertungsmaß. Dazu setzten wir greater_is_better=False.
    # Dementsprechend werden die Scores im Grid Search auch negativ sein und der Score, der am nähesten zu 0 ist der beste.

    from sklearn.metrics import make_scorer, mean_absolute_error
    # It takes a score function, such as
    # ~sklearn.metrics.accuracy_score,
    # ~sklearn.metrics.mean_squared_error,
    # ~sklearn.metrics.adjusted_rand_score or
    # ~sklearn.metrics.average_precision_score and returns a callable that ...

    # Score for KRR -> MAE
    score = make_scorer(mean_absolute_error, greater_is_better=False)
    return score


def plotPredictionAndData(pred_df, train_df, title):
    # Check dimensionality of data
    if len(train_df.columns) > 2: # 2D
        # define figure shape
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for df, color, label in (pred_df, "r", "Prediction"), (train_df, "C0", "Train"):
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
        for df, color, label in (pred_df, "r", "Prediction"), (train_df, "C0", "Train"):
            # Extract the x and y data from the dataframe
            x1 = df[df.columns[0]]
            y = df[df.columns[-1]]
            # plot x,y
            ax.plot(x1, y, color=color)
            ax.set_xlabel('x')
            ax.set_ylabel('y')

    plt.title(title)
    ax.legend()
    plt.show()
    return None


#TODO: move to model class
def saveModelToFile(filepath, model):
    # save the model to disk
    import joblib
    joblib.dump(model, filepath)
    print("\nModel saved to file:", filepath)
    return None

def loadModelFromFile(filepath, X_test, y_test):
    # load the model from disk
    import joblib
    print("\nLoad model from file:", filepath)
    loadedModel = joblib.load(filepath)
    testScore = loadedModel.score(X_test, y_test)
    return loadedModel, testScore


