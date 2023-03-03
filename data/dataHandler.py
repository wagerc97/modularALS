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


class DataHandler:
    def __init__(self, **kwargs):
        """ A class to handle data for ml model """
        # Data storage
        self.df = None                  # initial data
        self.train_df = None            # train split
        self.test_df = None             # test split
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.pred_df = None             # Dataframe with column name "y_pred"
        self.predXtest_df = None        # Dataframe with X_test and y_pred


    def setDataframe(self, df):
        self.df = df


    def kickStartFromDataframe(self, df=None):
        if self.df is None and df is None:
            raise ValueError("Please provide a dataframe for the dataHandler first")
        if self.df is None:
            self.df = df
        # Define data splits (train and test)
        self.defineDataSplits(self.df)


    def kickStartFromFile(self, filename=None, filepath=cfg.DATA_FILE, verbose=False):
        """ Read in data from file and define test- and trainsplits """
        # get data from CSV file
        _ = self.readDataFromCsvToDf(filename, filepath, verbose)

        # Define data splits (train and test)
        self.defineDataSplits(self.df)


    def defineDataSplits(self, param_df, random_seed=42):
        """
        Provide raw data saved in original df to Class.
        Split data into X_train, y_train, X_test, y_test (80/20 split)
        """
        self.df = param_df.copy()
        # Data split
        data_train, data_test = train_test_split(self.df, test_size=0.2, random_state=random_seed)

        # assign X all columns except "y" (which we want to predict)
        # assign y the "y" column
        X_train, y_train = data_train.drop(["y"], axis=1), data_train.y
        X_test, y_test = data_test.drop(["y"], axis=1), data_test.y

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Save train and test df (nice column names)
        self.train_df = self.X_train.assign(y=self.y_train)
        #print("\n\n----------------\n")
        #print(self.train_df)
        self.test_df = self.X_test.assign(y=self.y_test)
        #print("\n\n----------------\n")
        #print(self.test_df)


    def getDataSplits(self):
        return self.X_train, self.y_train, self.X_test, self.y_test, self.train_df, self.test_df


    def readDataFromCsvToDf(self, filename=None, filepath=None, verbose=False):
        """ Read in data training data from csv-file and save it as dataframe in variable. """
        if filepath is None and filename is None: # just default config
            filepath = cfg.DATA_FILE_NAME

        if filepath is None and filename is not None:
            filepath = os.path.join("..", filename)

        if filepath is not None and filename is not None:
            raise ValueError("Error: Either provide filename or whole filepath as argument.")

        # get data
        self.df = pd.read_csv(filepath, sep=';', header=1)
        # get problem name from first line
        with open(filepath, "r", newline='\n') as f:
            reader = csv.reader(f)
            problem_name = next(reader)[0]
        print("Found data for problem:", problem_name)
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
