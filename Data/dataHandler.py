#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Class for MachineLearning model which wraps the model functions and thus provides a clearn UI. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import csv
import pandas as pd
import myconfig as cfg     # project specific configurations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


class DataHandler:
    def __init__(self, seed=42, **kwargs):
        """ A class to handle data for ml model """
        self.split_seed = seed
        # For model training
        self.df = None                  # initial data
        self.train_df = None            # train split
        self.test_df = None             # test split
        self.X_train = None             # 80% of df, only columns x1, x2, ...
        self.y_train = None             # 80% of df, only column y (labels)
        self.X_test = None              # 20% of df, only columns x1, x2, ...
        self.y_test = None              # 20% of df, only column y (labels)
        self.y_pred_df = None           # Predicted labels - Dataframe with column name "y_pred"
        self.predXtest_df = None        # Dataframe with X_test and y_pred_df

        # for unseen data -> prediction
        self.X_df = None
        self.y_df = None


    def __del__(self):
        """ destructor frees up memory """
        print(f"---Object {self.__class__.__name__} destroyed")


    def setDataframe(self, df):
        self.df = df


    def kickStartFromDataframe(self, df=None):
        if self.df is None and df is None:
            raise ValueError("Please provide a dataframe for the dataHandler first")
        if self.df is None:
            self.df = df
        # Define data splits (train and test)
        self.splitDataForModelGeneration(self.df)


    def splitDataForPrediction(self, param_df=None, random_seed=None):
        """
        Split data into train and test dataframe.
        Raw data was first provided and is saved in original df in this Class object.
        :param param_df:
        :param random_seed:
        :return: randomized X and y splits dataframe of provided data
        """
        if param_df is None:
            param_df = self.df
        if random_seed is None:
            random_seed = self.split_seed

        # Save parameter dataframe and avoid manipulating it
        self.df = param_df
        df = self.df.copy()

        # Split Data: randomize rows
        df = shuffle(df)

        # Split X and y
        #   assign X all columns except label "y"
        #   assign y the "y" column
        self.X_df, self.y_df = df.drop(["y"], axis=1), df.y

        return self.X_df, self.y_df


    def splitDataForModelGeneration(self, param_df, random_seed=None):
        """
        Split data into X_train, y_train, X_test, y_test (80/20 split)
        Raw data was first provided and is saved in original df in this Class object.
        """
        if random_seed is None:
            random_seed = self.split_seed

        # Save parameter dataframe and avoid manipulating it
        self.df = param_df
        df = self.df.copy()

        # Split data 80/20, shuffle rows
        data_train, data_test = train_test_split(df, test_size=0.2, random_state=random_seed)

        # Split X and y
        #   assign X all columns except label "y"
        #   assign y the "y" column
        X_train, y_train = data_train.drop(["y"], axis=1), data_train.y
        X_test, y_test = data_test.drop(["y"], axis=1), data_test.y

        # Save splits in data handler
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Save train and test df (nice column names)
        self.train_df = self.X_train.assign(y=self.y_train)
        self.test_df = self.X_test.assign(y=self.y_test)


    def getDataSplits(self):
        """ :return: X_train, y_train, X_test, y_test, train_df, test_df """
        return self.X_train, self.y_train, self.X_test, self.y_test, self.train_df, self.test_df


    def readDataFromCsvToDf(self, filename=None, filepath=None, verbose=False):
        """ Read in data training data from csv-file and save it as dataframe in variable. """
        if verbose:
            print("filename:", filename)
            print("filepath:", filepath)

        if filepath is not None and filename is not None:
            raise ValueError("You have to provide a filename or an absolute filepath.")
        elif filepath is not None and filename is None:
            pass
        elif filepath is None and filename is not None:
            filepath = os.path.join("..", filename)
        else: # filepath is None and filename is None: # just default config
            #filepath = cfg.TRAIN_DATA_NAME
            raise ValueError("You have to provide a filename or an absolute filepath.")

        # get data
        self.df = pd.read_csv(filepath, sep=';', header=1)
        # get problem name from first line
        with open(filepath, "r", newline='\n') as f:
            reader = csv.reader(f)
            problem_name = next(reader)[0]
        print("\nFound data for problem:", problem_name)
        if verbose:
            print("csv filepath:", filepath)
            print(self.df)
        return problem_name


    # todo: check if transform only or fit as well for "fit_transform"
    def preprocessData(self, scaler_X, scaler_y):
        """ Preprocess data according to given scaler """
        # define preprocessor
        #print("\n\n", self.X_train)
        #print("\n\n", self.X_test)

        X_columns = []
        for i in range(self.X_train.shape[1]):  # get number of columns
            X_columns.append(f"x{i}")   # define column names

        scaler_X = StandardScaler()  # standardscaler__
        #self.scaler_y = StandardScaler()  # standardscaler__
        # fit scaler to train data and transform train data
        self.X_train = pd.DataFrame(scaler_X.fit_transform(self.X_train), columns=X_columns)
        #self.y_train = pd.DataFrame(scaler_y.fit_transform(self.y_train), columns=['y'])
        # transform test data
        self.X_test = pd.DataFrame(scaler_X.transform(self.X_test), columns=X_columns)
        #self.y_test = pd.DataFrame(scaler_y.transform(self.y_test), columns=['y'])
        #print("\n\n", self.X_train)
        #print("\n\n", self.X_test)

