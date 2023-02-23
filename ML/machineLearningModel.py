"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Class for ML model which wraps the model functions and thus provides a clearn UI. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import pandas as pd
import utility as util
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score


class MachineLearningModel:
    def __init__(self, model_type, normalize=False, **kwargs):
        """
        A class for building and training machine learning models using scikit-learn.

        :param model_type: The type of machine learning model to build. Currently, only linear regression is supported.
        Implemented: [linear_regression, lasso, ridge]
        :type model_type: str
        :param normalize: Whether to normalize the input features before training the model. Default is False.
        :type normalize: bool

        Example usage:
        >> model = MachineLearningModel('linear_regression', normalize=True)
        >> model.train(X_train, y_train)
        >> y_pred = model.predict(X_test)
        >> print(model)
        """
        self.scaler = None
        self.verbose = None
        self.normalize = normalize
        # Train data
        self.X_train = None
        self.y_train = None

        # Create model from input parameter
        if model_type == 'linear_regression':
            self.model = LinearRegression(**kwargs)
        elif model_type == 'lasso':
            self.model = Lasso(**kwargs)
        elif model_type == 'ridge':
            self.model = Ridge(**kwargs)
        else:
            raise ValueError(f'Invalid model type: {model_type}')

    def __str__(self):
        """ Allows printing information about the model """
        return f"{type(self.model).__name__} model with parameters {self.model.get_params()}"

    def train(self, X_train, y_train):
        """
        Train the machine learning model on a given training dataset.

        :param X_train: The input features for the training dataset.
        :type X_train: numpy.ndarray
        :param y_train: The target variable for the training dataset.
        :type y_train: numpy.ndarray

        Example usage:
        >> model.train(X_train, y_train)
        """
        self.X_train = X_train
        self.y_train = y_train
        if self.normalize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)

        self.model.fit(X_train, y_train)

    def predict(self, X_test, verbose=False):
        """
        Make predictions using the trained machine learning model.

        :param X_test: The input features for the test dataset.
        :type X_test: numpy.ndarray
        :return: The predicted target variable for the test dataset.
        :rtype: numpy.ndarray

        Example usage:
        >> y_pred = model.predict(X_test)
        """
        if self.normalize:
            X_test = self.scaler.transform(X_test)
        prediction = self.model.predict(X_test)
        if verbose:
            print("y_pred:\n", prediction)
        return pd.DataFrame(prediction, columns=["y_pred"])

    def plotPrediction(self, y_predict):
        # Fancy plot
        plt.figure(figsize=(9, 6))
        plt.plot(self.X_train, y_predict, "r-", linewidth=2, label="Predictions")
        plt.plot(self.X_train, self.y_train, "b.")
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$y$", rotation=0, fontsize=18)
        plt.legend(loc="upper left", fontsize=14)
        plt.title(f"Model: {self.model.__class__.__name__}")
        plt.axis([0, 2, 0, 15])
        util.save_fig("linear_model_predictions_plot")
        plt.show()

    def getTrainedParameters(self):
        """
        Returns Intercept and Coefficients. Can ONLY be called after training! """
        return self.model.intercept_[0], self.model.coef_[0]

    def printTrainedParameters(self):
        """ Print the intercept and the coefficients """
        print("intercept:\t\t", self.model.intercept_[0])
        print("coefficients:\t", self.model.coef_[0])

    def assessAccuracy(self, y_pred):
        """ Using sklearn.metrics.accuracy_score"""
        acc = accuracy_score(y_true=self.y_train, y_pred=y_pred, normalize=self.normalize)
        return acc

    def assessAccuracy2(self, theta_best):
        """ Returns absolut error of theta true minus predicted """
        predictedParameters = np.array([self.model.intercept_[0], self.model.coef_[0][0]])
        theta_best = np.array([item for sublist in theta_best for item in sublist])
        print("theta_best:", theta_best)
        print("predictedParameters:", predictedParameters)
        return theta_best - predictedParameters

