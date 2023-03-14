#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Class for MachineLearning model which wraps the model functions and thus provides a clearn UI. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from Data.dataHandler import DataHandler     # inherit the dataHandling class elements
import myconfig     # project specific configurations

# Ignore FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ModelGenerator(DataHandler):
    def __init__(self, **kwargs):
        """ A class for building and training machine learning models using scikit-learn. """

        super().__init__(**kwargs)      # initialize the class instance of dataHandler with extra parameters listed below
        self.normalize = None           # BOOLEAN value if data will be standardized
        self.scaler_X = None            # standardization method for X dataframe
        self.scaler_y = None            # standardization type for y dataframe
        self.model_type = None          # model object
        self.prefix = None              # model type defines suffix of hyperparameters
        self.pipeline = None            # preprocessor + model type
        self.scorer = None              # score function (MAE, MSE, ...)
        self.paramGrid = None           # hyperparameter grid for gridsearch
        self.grid_search_cv = None      # gridsearch with cv wrapping pipeline
        self.gscv_test_scores = None    # cross-validation test scores of gridsearch
        self.testScore = None           # test score of best estimator
        self.bestEstimator = None       # best estimator saved after training

        # Results
        self.train_result_df = None     # ordered train results
        self.tr_df_slim = None          # TestResult_DataFrame: Only interesting columns, first 5 rows, rounded to 2 decimals

        print(f"New {self.__class__.__name__} object created ")


    def __str__(self):
        """ Allows printing information about the model """
        try:
            return f"{type(self.model_type).__name__} model with parameters {self.model_type.get_params()}"
        except:
            return f"Chosen model: {type(self.model_type).__name__}"


    def __del__(self):
        """ destructor frees up memory """
        print(f"\nObject {self.__class__.__name__} destroyed")


    #TODO: different scorers for hyperparameter search and training
    def defineScorer(self, score_type):
        # It takes a score function, such as
        # ~sklearn.metrics.accuracy_score,
        # ~sklearn.metrics.mean_squared_error,
        # ~sklearn.metrics.adjusted_rand_score or
        # ~sklearn.metrics.average_precision_score and returns a callable that ...

        if score_type.lower() in ('mae', 'mean_absolute_error'):
            self.scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        elif score_type.lower() in ('accuracy_score', 'acc'):
            self.scorer = make_scorer(accuracy_score)
        elif score_type.lower() in ('root_mean_squared_error', 'rmse'): # higher weights for outliers
            self.scorer = make_scorer(mean_squared_error) # TODO: needs root aswell (=RMSE)
        # todo: R² is also interesting
        else:
            raise ValueError(f'Invalid score type: {score_type}')


    def createPipeline(self, model_name, normalize=False):
        """
        Create Pipeline according to input parameters
        - preprocessor
        - model type
        """
        # Create model from input parameter
        if model_name.lower() in ('krr', 'kernel_ridge_regressor'):
            #self.model_type = KernelRidge(kernel="rbf") # kernelridge__
            self.model_type = KernelRidge() # kernelridge__
        elif model_name.lower() in ('svr', 'support_vector_regression'):
            #self.model_type = SVR(kernel="rbf")
            self.model_type = SVR()
        else:
            raise ValueError(f'Invalid model type: "{model_name}"')
        print("model_type:", self.model_type, "\n")

        # define preprocessing, scaler, encoder, ...
        self.normalize = normalize
        print("Normalize data:", self.normalize)
        if self.normalize:
            self.preprocessData()

        # create pipeline
        self.pipeline = make_pipeline(self.scaler_X, self.model_type)
        # save model suffix for hyperparameters
        self.prefix = str(self.pipeline).split("'")[-2] + "__"


    def getPipeline(self):
        return self.pipeline


    def getModelSuffix(self):
        return self.prefix


    def defineParamGrid(self, param_grid):
        """ Define hyperparameters for grid search """
        if param_grid is None:
            # my default parameters for krr
            print(">> choosing default KRR parameters because no param grid was given")
            param_grid = [{
                "alpha": [0.001, 0.01, 0.1, 1],
                "gamma": [0.001, 0.01, 0.03, 0.05, 0.1],
                "kernel": ["rbf"]
            }]

        # Add prefix to keys
        for key in list(param_grid.keys()):
            new_key = self.prefix + key
            param_grid[new_key] = param_grid.pop(key)

        self.paramGrid = param_grid


    def applyGridSearchCV(self, verbose=False):
        """ Exhaustively search for best hyperparameters with GridSearchCV """
        # https://www.kaggle.com/code/wagerc97/uebung1-bsp2-angabe

        grid_search = GridSearchCV(
            estimator=self.pipeline,    # fresh estimator
            param_grid=self.paramGrid,  # grid of hyperparameters
            n_jobs=-1,                  # jobs to run in parallel (-1 uses all processes available)
            scoring=self.scorer,        # using a callable to score each model
            cv=5,                       # k-fold cross validation (default=5)
            verbose=1
        )
        self.grid_search_cv = grid_search

        # TRAIN: fit the newly established model with data
        print("\nStart training model:")
        self.grid_search_cv.fit(self.X_train, self.y_train)
        #self.grid_search_cv.fit(self.X_train.values, self.y_train)  #TODO: is using the values from the df without feature names OK?

        # TEST: Cross-validation of gridsearch with TEST data
        self.gscv_test_scores = cross_val_score(self.grid_search_cv, self.X_test, self.y_test)

        # save the best estimator in class attribute
        self.bestEstimator = self.grid_search_cv.best_estimator_

        # table of k cross validation results
        self.train_result_df = pd.DataFrame(grid_search.cv_results_).sort_values("rank_test_score")

        # Only interesting columns, rounded to 2 decimals, and only the first 5 rows, shorter header names
        #self.tr_df_slim = self.train_result_df.drop(self.train_result_df.filter(regex='split|params|std_fit_time|std_score_time').columns, axis=1).head().round(decimals=2)
        self.simplifyResultDf()

        if verbose:
            self.printSimplifiedTable(self.tr_df_slim)


    def simplifyResultDf(self):
        """ Adapt and save slimmer version of result-dataframe
        -> Only interesting columns, rounded to 2 decimals, and only the first 5 rows, shorter header names
        self.train_result_df is the full result df
        self.tr_df_slim is the simplified version this function creates
        :return:
        """
        ptable = self.train_result_df
        # filter out columns
        ptable = ptable.drop(ptable.filter(regex='split|params|std_fit_time|std_score_time').columns, axis=1)
        # sort and show only head
        ptable = ptable.sort_values("rank_test_score").head()
        # round results
        ptable = ptable.round(decimals=2)
        # replace words in header with regex
        # HERE we will ignore the FutureWarning
        ptable.columns = ptable.columns.str.replace(rf"(param_|{self.prefix})", "")
        self.tr_df_slim = ptable


    def printSimplifiedTable(self, ptable):
        """ Print simplified table of train results for model """
        print("\n+--------------------------------------------------------+")
        print("Number of columns in table:", len(ptable))
        print("==========================================================")
        # print all columns
        pd.set_option('display.max_columns', None)
        self.simplifyResultDf()
        print(ptable)
        print("+--------------------------------------------------------+\n")


    def saveDfToTemporaryFile(self, df=None):
        """ Save the most important train results in a separate file  """
        dirpath = myconfig.TMP_TABLE_DIR
        if df is None:
            df = self.tr_df_slim
            filepath = myconfig.TMP_TABLE_FILE
        else:
            filepath = os.path.join(dirpath, "newDataframe.csv")
        # assert folder exists
        os.makedirs(dirpath, exist_ok=True)
        # Write dataframe to CSV file and overwrite existing file
        df.to_csv(filepath, index=False, mode='w', sep=';')
        print("\nResults were saved to file:", filepath)


    def getTestScore(self, X_test=None, y_test=None):
        """
        Get the prediction score (R²) of the best estimator on the TEST dataset.
        Also compute and store the test score in the object, if it was not already done.
        Input of X_test, y_test is optional. Default is saved in class already.
        """
        if self.testScore is None:
            if X_test is None:
                X_test = self.X_test
            if y_test is None:
                y_test = self.y_test
            bestModel_testScore = self.bestEstimator.score(X_test, y_test)
            self.testScore = bestModel_testScore
            #bestModel_testScore = r2_score(y_true=y_test, y_pred=self.y_pred_df)
            return round(bestModel_testScore, 3)
        else:
            return round(self.testScore, 3)


    def getAllCvTestScores(self):
        """ Get the all cross-validated TEST scores of the grid_search object """
        return np.round(self.gscv_test_scores, 3)


    def getMeanTestScoreOfGridsearchCV(self):
        """ Get the mean across all cross-validated TEST scores of the grid_search object """
        return np.round(np.mean(self.gscv_test_scores), 3)


    def getBestEstimator(self):
        """ Get the prediction score of the best estimator on the TEST dataset """
        return self.bestEstimator


    def predict(self, X_test=None, verbose=False):
        """
        Make predictions using the trained machine learning model.
        Input of X_test is optional. Default is saved in Class already.
        """
        if X_test is None:
            X_test = self.X_test

        # Predict on dataset
        y_pred = self.grid_search_cv.best_estimator_.predict(X_test)
        if self.normalize:
            scaler = self.scaler_X
            y_pred = pd.DataFrame(scaler.inverse_transform(y_pred), columns=['y'])  # transform back
        self.y_pred_df = pd.DataFrame(y_pred, columns=["y_pred"])

        # Save combined dataframe for later plotting
        X_test_copy = self.X_test.copy().reset_index(drop=True)  # reset index of this copy only which was shuffled for model training
        self.predXtest_df = X_test_copy.assign(y=self.y_pred_df)

        if verbose:
            print("y_pred_df:\n", self.y_pred_df)
        return self.y_pred_df, self.predXtest_df


    def saveModelToFile(self, filepath=myconfig.MODEL_FILE):
        """save the model to disk"""
        # make sure the file destination exists
        os.makedirs(myconfig.MODEL_DIR, exist_ok=True)  # Make sure the directory exists
        joblib.dump(self.bestEstimator, filepath)
        print("\nModel saved to file:", filepath)
        return None

    @staticmethod
    def loadModelFromFile(filepath=myconfig.MODEL_FILE):
        """load the model from disk"""
        print("\nLoad model from file:", filepath)
        loadedModel = joblib.load(filepath)
        return loadedModel


    def plotPredictionAndTrainData(self, title):
        """
        Plot predicted data against X_test data and train_df as well.
        """
        # Check if prediction exists
        if self.predXtest_df is None:
            self.predict()
        # Check dimensionality of data
        if len(self.train_df.columns) == 3:  # 2D
            # define figure shape
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for df, color, label in (
                    self.predXtest_df, "r", "Prediction" + f" (n={self.predXtest_df.shape[0]})"), \
                    (self.train_df, "C0", "Train" + f" (n={self.train_df.shape[0]})"):
                # Extract the x and y data from the dataframe
                x1 = df[df.columns[0]]
                x2 = df[df.columns[1]]
                y = df[df.columns[-1]]
                # plot x,y
                ax.scatter(x1, x2, y, color=color, label=label)
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_zlabel('y')
        else:
            raise ValueError("Can only plot df with 3 columns (x1, x2, y)")

        title = title + f" (n={self.predXtest_df.shape[0] + self.train_df.shape[0]})"
        #title = title + f" Test score={self.getTestScore()}"
        title = title + f" Test score={self.getTestScore()}"
        plt.title(title)
        ax.legend()
        plt.show()



