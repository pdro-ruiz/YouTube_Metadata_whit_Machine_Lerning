"""
utf-8
lightgbm_regressor.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module contains the implementation of the LightGBM model for regression tasks.

Classes:
- LightGBMRegressorModel: encapsulates the model's logic.

Main methods:
- __init__: Initializes the model with the specified configuration.
- fit: Trains the model.
- predict: Makes predictions.
- save: Saves the trained model.
- load: Loads the trained model.

The model uses a configuration file (config.conf) to specify hyperparameters, data paths, and other relevant options.
"""

import os
import pickle
import configparser
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import logging


def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)


class LightGBMRegressorModel:
    def __init__(self, config_path, config_section='lightgbm_reg_two', logger=None):
        """
        Initializes the LightGBM Regressor model with the specified configuration section.

        Parameters:
            config_path (str): Path to the configuration file.
            config_section (str): Section in the configuration file for this model.
            logger (logging.Logger, optional): Logger instance to record events.
            
        """
        self.logger = logger or logging.getLogger(__name__)
        
        config = configparser.ConfigParser()
        config.read(config_path)

        # Check configuration
        if config_section not in config:
            raise ValueError(f"Section '{config_section}' not found.")

        # LightGBM Configuration
        self.model = LGBMRegressor(
            learning_rate=config.getfloat(config_section, 'learning_rate'),
            n_estimators=config.getint(config_section, 'n_estimators'),
            random_state=config.getint(config_section, 'random_state')
        )

        # Features and target
        self.features = [feature.strip() for feature in config.get(config_section, 'features').split(',')]
        self.target = config.get(config_section, 'target')
        
        # Paths for X_test and y_test
        eval_section = f"{config_section}_eval"
        
        self.X_test_path = config[eval_section]['X_test']
        self.y_test_path = config[eval_section]['y_test']

        self.logger.info(f"LightGBMRegressorModel initialized with '{config_section}'.")


    def __preprocess_data(self, df):
        """
        Preprocesses the data.

        Parameters:
            df (pd.DataFrame)

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        self.logger.debug("Preparing data for LightGBM.")
        X = df[self.features]
        y = df[self.target]
        return train_test_split(X, y, test_size=0.2, random_state=42)


    def predict(self, X):
        """
        Makes predictions.

        Parameters:
            X (pd.DataFrame or numpy array): Input data.

        Returns:
            array: Predicted values.
        """
        self.logger.debug("Making predictions with LightGBM.")
        return self.model.predict(X)


    def fit(self, df):
        """
        Trains the model.

        Parameters:
            df (pd.DataFrame)
            
        Returns:
            tuple: (X_test, y_test, metrics)
        """
        self.logger.info("Training LightGBM.")
        X_train, X_test, y_train, y_test = self.__preprocess_data(df)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'r2': r2_score(y_test, y_pred)
        }

        self.logger.info(f"Training completed. Metrics: {metrics}")
        return X_test, y_test, metrics


    def save(self, model_path):
        """
        Saves the model.

        Parameters:
            model_path (str): Path to save the model.
        """
        self.logger.info(f"Saving LightGBM model to {model_path}.")
        ensure_dir_exists(os.path.dirname(model_path))
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        self.logger.debug("LightGBM model saved successfully.")


    def load(self, model_path):
        """
        Loads a model.

        Parameters:
            model_path (str): Path to the model file.
        """
        self.logger.info(f"Loading LightGBM model from {model_path}.")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.logger.debug("LightGBM model loaded successfully.")
